"""RunPod backend — submits a Pod, waits for it to finish, returns artifact URIs.

Only ``train`` is GPU-bound and runs on RunPod. The other ops delegate to the
local backend; doing so keeps prepare/publish/project/walk/eval cheap and lets
the train pod's published checkpoint flow back through the same registry.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from deepVogue.notifications import slack

log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
TERMINAL_STATUSES = {"EXITED", "TERMINATED", "FAILED"}
SUCCESS_STATUSES = {"EXITED"}


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} must be set for runpod backend")
    return val


def _build_env(
    *,
    dataset_uri: str,
    run_uri: str,
    model_id: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int,
    resume_from: str | None,
) -> dict[str, str]:
    env = {
        "DV_DATASET_URI": dataset_uri,
        "DV_RUN_URI": run_uri,
        "DV_MODEL_ID": model_id,
        "DV_CFG": cfg,
        "DV_KIMG": str(kimg),
        "DV_GAMMA": str(gamma),
        "DV_BATCH": str(batch),
        "DV_RES": str(res),
    }
    if resume_from:
        env["DV_RESUME_FROM"] = resume_from
    # Pass through Mlflow + publish target + Slack so the pod can log + register.
    for k in (
        "MLFLOW_TRACKING_URI",
        "DV_PUBLISH_TARGET",
        "DV_ARTIFACT_BACKEND",
        "DV_MODELS_ROOT",
        "DV_MODELS_YAML",
        "SLACK_WEBHOOK_URL",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    ):
        v = os.environ.get(k)
        if v:
            env[k] = v
    return env


def _submit(env: dict[str, str], name: str) -> str:
    """Create a RunPod GPU pod, return its id."""
    import runpod  # imported lazily; SDK only required when backend is used

    runpod.api_key = _require_env("RUNPOD_API_KEY")
    image = _require_env("RUNPOD_IMAGE")
    gpu_type = os.environ.get("RUNPOD_GPU_TYPE", "NVIDIA H100 80GB HBM3")
    volume_gb = int(os.environ.get("RUNPOD_VOLUME_GB", "50"))

    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=volume_gb,
        container_disk_in_gb=20,
        env=env,
        cloud_type="SECURE",
    )
    pod_id = pod["id"] if isinstance(pod, dict) else pod.id
    log.info("runpod: submitted pod %s (image=%s gpu=%s)", pod_id, image, gpu_type)
    return pod_id


def _wait(pod_id: str) -> dict[str, Any]:
    """Block until pod reaches a terminal status; return final pod dict."""
    import runpod

    while True:
        info = runpod.get_pod(pod_id)
        status = (info or {}).get("desiredStatus") or (info or {}).get("status") or ""
        log.info("runpod: pod %s status=%s", pod_id, status)
        if status.upper() in TERMINAL_STATUSES:
            return info or {}
        time.sleep(POLL_INTERVAL_SECONDS)


def train(
    *,
    dataset_name: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int = 256,
    target_uri: str | None = None,
    resume_from: str | None = None,
    **_,
) -> dict[str, Any]:
    """Submit a RunPod training job and return the published checkpoint URI.

    Reads dataset location from ``DV_DATASET_URI`` (gs://...) if ``target_uri``
    isn't a gs:// URI — train inputs and run outputs always live on GCS for the
    RunPod backend.
    """
    dataset_uri = os.environ.get("DV_DATASET_URI") or _require_env("DV_DATASET_URI")
    run_uri = target_uri or os.environ.get("DV_RUN_URI") or _require_env("DV_RUN_URI")
    model_id = os.environ.get("DV_MODEL_ID", dataset_name)

    env = _build_env(
        dataset_uri=dataset_uri,
        run_uri=run_uri,
        model_id=model_id,
        cfg=cfg,
        kimg=kimg,
        gamma=gamma,
        batch=batch,
        res=res,
        resume_from=resume_from,
    )

    pod_name = f"dv-{model_id}-{int(time.time())}"
    pod_id = _submit(env, pod_name)
    slack.notify_event(
        "runpod",
        "info",
        f"pod {pod_id} submitted for {model_id}",
        {"image": env.get("RUNPOD_IMAGE", "")},
    )

    final = _wait(pod_id)
    status = (final.get("desiredStatus") or final.get("status") or "").upper()

    # Best-effort: stop the pod so it stops billing.
    try:
        import runpod

        runpod.stop_pod(pod_id)
    except Exception as e:
        log.warning("runpod: stop_pod %s failed: %s", pod_id, e)

    if status not in SUCCESS_STATUSES:
        slack.notify_failure("runpod", f"pod {pod_id} finished with status={status}")
        raise RuntimeError(f"runpod pod {pod_id} failed (status={status})")

    pkl_uri = f"{run_uri.rstrip('/')}/network-snapshot-{kimg:06d}.pkl"
    return {"pkl": pkl_uri, "kimg": kimg, "pod_id": pod_id, "fid": None}


# ----- delegate non-GPU ops to the local backend -----


def _delegate(op_name: str, **kw):
    from . import local

    log.info("runpod.%s: delegating to local backend", op_name)
    return getattr(local, op_name)(**kw)


def prepare(**kw):
    return _delegate("prepare", **kw)


def publish(**kw):
    return _delegate("publish", **kw)


def project(**kw):
    return _delegate("project", **kw)


def walk(**kw):
    return _delegate("walk", **kw)


def eval(**kw):
    return _delegate("eval", **kw)
