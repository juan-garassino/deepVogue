"""Vertex AI backend — submits a CustomJob, waits for completion, returns artifact URIs.

Only ``train`` is GPU-bound and runs on Vertex. The other ops delegate to the
local backend, same as the RunPod backend.

Lifecycle: Vertex AI provisions a GPU VM for the job, runs the train container
to completion, and releases the VM automatically — there is no idle cost and
no self-terminate plumbing (unlike RunPod). Auth inside the container is
ambient ADC via the attached service account, so no SA JSON key is involved;
the entrypoint detects the missing GOOGLE_APPLICATION_CREDENTIALS_JSON and
falls back to the metadata server.
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

DEFAULT_MACHINE_TYPE = "g2-standard-8"
DEFAULT_ACCELERATOR = "NVIDIA_L4"
DEFAULT_BOOT_DISK_GB = 200


def _submit(env: dict[str, str], name: str, max_hours: float):
    """Create and submit a Vertex AI CustomJob, return the job object."""
    from google.cloud import aiplatform

    project = require_env("GCP_PROJECT", "vertex")
    region = os.environ.get("GCP_REGION", "europe-west1")
    image = os.environ.get("VERTEX_IMAGE") or require_env("RUNPOD_IMAGE", "vertex")
    machine_type = os.environ.get("VERTEX_MACHINE_TYPE", DEFAULT_MACHINE_TYPE)
    accelerator = os.environ.get("VERTEX_ACCELERATOR", DEFAULT_ACCELERATOR)
    accelerator_count = int(os.environ.get("VERTEX_ACCELERATOR_COUNT", "1"))
    boot_disk_gb = int(os.environ.get("VERTEX_BOOT_DISK_GB", str(DEFAULT_BOOT_DISK_GB)))
    service_account = os.environ.get(
        "VERTEX_SERVICE_ACCOUNT",
        f"deepvogue-trainer-sa@{project}.iam.gserviceaccount.com",
    )

    aiplatform.init(project=project, location=region)
    job = aiplatform.CustomJob(
        display_name=name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": accelerator,
                    "accelerator_count": accelerator_count,
                },
                "replica_count": 1,
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": boot_disk_gb,
                },
                "container_spec": {
                    "image_uri": image,
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                },
            }
        ],
    )
    job.submit(service_account=service_account, timeout=int(max_hours * 3600))
    log.info(
        "vertex: submitted %s (image=%s machine=%s accel=%sx%d)",
        job.resource_name,
        image,
        machine_type,
        accelerator,
        accelerator_count,
    )
    return job


def _job_state_name(job) -> str:
    state = getattr(job, "state", None)
    return getattr(state, "name", str(state) or "UNKNOWN")


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
    """Submit a Vertex AI training job and return the published checkpoint URI.

    Required env (launcher side):
      GCP_PROJECT, VERTEX_IMAGE (or RUNPOD_IMAGE — same container, but it must
      be pullable by Vertex, i.e. pushed to Artifact Registry),
      DV_DATASET_URI, DV_RUN_URI (or target_uri), DV_PUBLISH_TARGET
    Optional:
      GCP_REGION (default europe-west1), VERTEX_MACHINE_TYPE (g2-standard-8),
      VERTEX_ACCELERATOR (NVIDIA_L4; use NVIDIA_TESLA_T4 + n1-standard-8 for
      the cheap tier), VERTEX_ACCELERATOR_COUNT, VERTEX_BOOT_DISK_GB,
      VERTEX_SERVICE_ACCOUNT (default deepvogue-trainer-sa@$GCP_PROJECT...),
      VERTEX_MAX_TRAIN_HOURS (default 24), DV_MODEL_ID, DV_RESUME_FROM
    """
    dataset_uri = require_env("DV_DATASET_URI", "vertex")
    run_uri = target_uri or require_env("DV_RUN_URI", "vertex")
    model_id = os.environ.get("DV_MODEL_ID", dataset_name)
    max_hours = float(os.environ.get("VERTEX_MAX_TRAIN_HOURS", "24"))

    extra: dict[str, str] = {}
    if os.environ.get("DV_FAKE_TRAIN"):
        # opt-in passthrough for CI smokes; loud because a leaked =1 on a real
        # run produces a stub pkl and phantom success
        log.warning(
            "forwarding DV_FAKE_TRAIN=%s into the job", os.environ["DV_FAKE_TRAIN"]
        )
        extra["DV_FAKE_TRAIN"] = os.environ["DV_FAKE_TRAIN"]

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
        extra=extra,
    )

    job_name = f"dv-{model_id}-{int(time.time())}"
    job = _submit(env, job_name, max_hours)

    # Surface the job id immediately so a Ctrl-C'd launcher can recover it.
    sys.stdout.write(
        json.dumps({"event": "submitted", "job": job.resource_name}) + "\n"
    )
    sys.stdout.flush()
    slack.notify_event(
        "vertex",
        "info",
        f"job {job.resource_name} submitted for {model_id}",
        {
            "image": os.environ.get("VERTEX_IMAGE")
            or os.environ.get("RUNPOD_IMAGE", ""),
            "max_hours": str(max_hours),
        },
    )

    start = time.time()
    failure: Exception | None = None
    try:
        # Blocks until the job reaches a terminal state; Vertex releases the
        # GPU VM itself — no reaper needed. Raises on job failure.
        job.wait()
    except Exception as e:  # SDK raises RuntimeError on JOB_STATE_FAILED
        failure = e
    elapsed = int(time.time() - start)
    state = _job_state_name(job)

    pkl_uri = snapshot_uri(run_uri, kimg)

    if failure is None and state.endswith("SUCCEEDED"):
        slack.notify_success(
            "vertex",
            f"job done ({state}) for {model_id}",
            {"pkl": pkl_uri, "kimg": str(kimg), "elapsed_s": str(elapsed)},
        )
        return {
            "pkl": pkl_uri,
            "kimg": kimg,
            "job": job.resource_name,
            "fid": None,
            "elapsed_s": elapsed,
        }

    slack.notify_failure(
        "vertex",
        f"job {state.lower()} for {model_id}",
        {"elapsed_s": str(elapsed), "error": str(failure) if failure else ""},
    )
    raise RuntimeError(
        f"vertex job {job.resource_name} ended with state={state}"
    ) from failure


# ----- delegate non-GPU ops to the local backend -----


def _delegate(op_name: str, **kw):
    return delegate_to_local("vertex", op_name, **kw)


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
