"""Shared helpers for GPU train backends (runpod, vertex).

The cross-project "copy, don't abstract" rule applies between repos; inside
this package the submit backends share their env-contract plumbing here.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Optional launcher env forwarded into every train container when set.
_PASSTHROUGH_KEYS = (
    "MLFLOW_TRACKING_URI",
    "DV_PUBLISH_TARGET",
    "DV_ARTIFACT_BACKEND",
    "DV_MODELS_ROOT",
    "DV_MODELS_YAML",
    "SLACK_WEBHOOK_URL",
    "DV_FAKE_TRAIN",
)


def require_env(name: str, backend: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} must be set for {backend} backend")
    return val


def build_train_env(
    *,
    dataset_uri: str,
    run_uri: str,
    model_id: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int,
    resume_from: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The DV_* env contract the train entrypoint validates, plus passthrough.

    ``extra`` carries backend-specific keys (e.g. RunPod's API key + SA JSON;
    Vertex needs none — ambient ADC).
    """
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
    for k in _PASSTHROUGH_KEYS:
        v = os.environ.get(k)
        if v:
            env[k] = v
    if extra:
        env.update(extra)
    return env


def snapshot_uri(run_uri: str, kimg: int) -> str:
    return f"{run_uri.rstrip('/')}/network-snapshot-{kimg:06d}.pkl"


def delegate_to_local(backend: str, op_name: str, **kw):
    from . import local

    log.info("%s.%s: delegating to local backend", backend, op_name)
    return getattr(local, op_name)(**kw)
