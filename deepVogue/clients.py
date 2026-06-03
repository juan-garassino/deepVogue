"""fsspec filesystem factory and URI resolution for deepVogue artifacts."""

from __future__ import annotations

import os
from typing import Any

import fsspec

_SUPPORTED = {"s3", "gcs", "memory", "file"}


def get_artifact_fs() -> Any:
    backend = os.environ.get("DV_ARTIFACT_BACKEND", "file").lower()
    if backend not in _SUPPORTED:
        raise ValueError(
            f"unknown DV_ARTIFACT_BACKEND={backend!r}; expected one of {_SUPPORTED}"
        )
    if backend == "s3":
        endpoint = os.environ.get("DV_S3_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return fsspec.filesystem("s3", client_kwargs=client_kwargs)
    if backend == "gcs":
        return fsspec.filesystem("gcs")
    if backend == "memory":
        return fsspec.filesystem("memory")
    return fsspec.filesystem("file")


def resolve_uri(uri: str) -> str:
    """Pass absolute URIs through; resolve relative paths against DV_MODELS_ROOT."""
    if "://" in uri or uri.startswith("/"):
        return uri
    root = os.environ.get("DV_MODELS_ROOT", "").rstrip("/")
    if not root:
        return uri
    return f"{root}/{uri}"
