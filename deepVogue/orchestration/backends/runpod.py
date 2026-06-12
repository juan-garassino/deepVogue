"""RunPod backend — submits a Pod, waits for it to self-terminate, returns artifact URIs.

Only ``train`` is GPU-bound and runs on RunPod. The other ops delegate to the
local backend; doing so keeps prepare/publish/project/walk/eval cheap and lets
the train pod's published checkpoint flow back through the same registry.

Lifecycle: the pod's entrypoint calls RunPod's GraphQL ``podTerminate`` on exit
(success or failure). The orchestrator polls ``get_pod`` and treats either a
``TERMINATED`` status or a None/missing pod as success. As a guard we call
``terminate_pod`` ourselves at the end — idempotent if the pod is already gone.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from deepVogue.notifications import slack
from deepVogue.orchestration.backends._common import (
    build_train_env,
    delegate_to_local,
    require_env,
    snapshot_uri,
)

log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
SUCCESS_STATUSES = {"TERMINATED", "EXITED"}
FAILURE_STATUSES = {"FAILED", "DEAD"}


def _submit(env: dict[str, str], name: str) -> str:
    """Create a RunPod GPU pod, return its id."""
    import runpod

    runpod.api_key = require_env("RUNPOD_API_KEY", "runpod")
    image = require_env("RUNPOD_IMAGE", "runpod")
    gpu_type = os.environ.get("RUNPOD_GPU_TYPE", "NVIDIA H100 80GB HBM3")
    volume_gb = int(os.environ.get("RUNPOD_VOLUME_GB", "0"))
    container_disk_gb = int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "100"))

    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=volume_gb,
        container_disk_in_gb=container_disk_gb,
        env=env,
        cloud_type="SECURE",
    )
    pod_id = pod["id"] if isinstance(pod, dict) else pod.id
    log.info("runpod: submitted pod %s (image=%s gpu=%s)", pod_id, image, gpu_type)
    return pod_id


def _wait(pod_id: str, max_seconds: float) -> tuple[str, dict[str, Any] | None]:
    """Block until pod reaches a terminal state or max_seconds elapses.

    Returns ``(status, final_info)``. ``status`` is one of:
      - "TERMINATED" / "EXITED" — clean self-terminate by the entrypoint
      - "FAILED" / "DEAD"       — RunPod reported a failure status
      - "GONE"                  — get_pod returned None (pod removed)
      - "TIMEOUT"               — exceeded max_seconds
    """
    import runpod

    deadline = time.time() + max_seconds
    while True:
        info = runpod.get_pod(pod_id)
        if info is None:
            return "GONE", None
        status = (info.get("desiredStatus") or info.get("status") or "").upper()
        log.info("runpod: pod %s status=%s", pod_id, status)
        if status in SUCCESS_STATUSES:
            return status, info
        if status in FAILURE_STATUSES:
            return status, info
        if time.time() >= deadline:
            return "TIMEOUT", info
        time.sleep(POLL_INTERVAL_SECONDS)


def _terminate_quiet(pod_id: str) -> None:
    """Idempotent terminate; logs but doesn't raise on failure."""
    try:
        import runpod

        runpod.terminate_pod(pod_id)
        log.info("runpod: terminated pod %s", pod_id)
    except Exception as e:
        log.warning("runpod: terminate_pod %s failed: %s", pod_id, e)


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

    Required env (launcher side):
      RUNPOD_API_KEY, RUNPOD_IMAGE, DV_DATASET_URI, DV_RUN_URI (or target_uri),
      DV_PUBLISH_TARGET, GOOGLE_APPLICATION_CREDENTIALS_JSON
    Optional:
      RUNPOD_GPU_TYPE, RUNPOD_VOLUME_GB, RUNPOD_CONTAINER_DISK_GB,
      RUNPOD_MAX_TRAIN_HOURS (default 24), DV_MODEL_ID, DV_RESUME_FROM
    """
    dataset_uri = require_env("DV_DATASET_URI", "runpod")
    run_uri = target_uri or require_env("DV_RUN_URI", "runpod")
    model_id = os.environ.get("DV_MODEL_ID", dataset_name)
    max_hours = float(os.environ.get("RUNPOD_MAX_TRAIN_HOURS", "24"))

    env = build_train_env(
        dataset_uri=dataset_uri,
        run_uri=run_uri,
        model_id=model_id,
        cfg=cfg,
        kimg=kimg,
        gamma=gamma,
        batch=batch,
        res=res,
        resume_from=resume_from,
        extra={
            # the pod self-terminates via GraphQL and gsutil needs the SA key
            "RUNPOD_API_KEY": require_env("RUNPOD_API_KEY", "runpod"),
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": require_env(
                "GOOGLE_APPLICATION_CREDENTIALS_JSON", "runpod"
            ),
        },
    )

    pod_name = f"dv-{model_id}-{int(time.time())}"
    pod_id = _submit(env, pod_name)

    # Surface pod_id immediately so a Ctrl-C'd launcher can recover it.
    sys.stdout.write(json.dumps({"event": "submitted", "pod_id": pod_id}) + "\n")
    sys.stdout.flush()
    slack.notify_event(
        "runpod",
        "info",
        f"pod {pod_id} submitted for {model_id}",
        {"image": os.environ.get("RUNPOD_IMAGE", ""), "max_hours": str(max_hours)},
    )

    start = time.time()
    status, _info = _wait(pod_id, max_hours * 3600)
    elapsed = int(time.time() - start)

    # Guard terminate (idempotent; cheap if pod is gone).
    _terminate_quiet(pod_id)

    pkl_uri = snapshot_uri(run_uri, kimg)

    if status in SUCCESS_STATUSES or status == "GONE":
        slack.notify_success(
            "runpod",
            f"pod {pod_id} done ({status}) for {model_id}",
            {"pkl": pkl_uri, "kimg": str(kimg), "elapsed_s": str(elapsed)},
        )
        return {
            "pkl": pkl_uri,
            "kimg": kimg,
            "pod_id": pod_id,
            "fid": None,
            "elapsed_s": elapsed,
        }

    slack.notify_failure(
        "runpod",
        f"pod {pod_id} {status.lower()} for {model_id}",
        {"elapsed_s": str(elapsed), "max_hours": str(max_hours)},
    )
    raise RuntimeError(f"runpod pod {pod_id} ended with status={status}")


# ----- delegate non-GPU ops to the local backend -----


def _delegate(op_name: str, **kw):
    return delegate_to_local("runpod", op_name, **kw)


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
