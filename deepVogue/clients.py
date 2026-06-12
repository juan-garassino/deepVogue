"""fsspec filesystem factory and URI resolution for deepVogue artifacts."""

from __future__ import annotations

import os
from typing import Any

import fsspec

# Maps DV_ARTIFACT_BACKEND value to its URI scheme prefix (empty for local fs).
_BACKEND_SCHEMES = {"s3": "s3://", "gcs": "gs://", "memory": "memory://", "file": ""}


def _backend() -> str:
    backend = os.environ.get("DV_ARTIFACT_BACKEND", "file").lower()
    if backend not in _BACKEND_SCHEMES:
        raise ValueError(
            f"unknown DV_ARTIFACT_BACKEND={backend!r}; "
            f"expected one of {set(_BACKEND_SCHEMES)}"
        )
    return backend


def get_artifact_fs() -> Any:
    backend = _backend()
    if backend == "s3":
        endpoint = os.environ.get("DV_S3_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return fsspec.filesystem("s3", client_kwargs=client_kwargs)
    return fsspec.filesystem(backend)


def resolve_uri(uri: str) -> str:
    """Pass absolute URIs through; resolve relative paths against DV_MODELS_ROOT."""
    if "://" in uri or uri.startswith("/"):
        return uri
    root = os.environ.get("DV_MODELS_ROOT", "").rstrip("/")
    if not root:
        return uri
    return f"{root}/{uri}"


def strip_scheme(uri: str) -> str:
    """Return the URI without any ``<scheme>://`` prefix; pass through bare paths."""
    return uri.split("://", 1)[1] if "://" in uri else uri


def scheme_of(uri: str) -> str:
    """Return the URI scheme, or ``"file"`` for bare/absolute paths."""
    return uri.split("://", 1)[0] if "://" in uri else "file"


def split_uri(uri: str) -> tuple[str, str]:
    """Normalize a fsspec-style URI and return ``(uri_no_trailing_slash, fs_path)``.

    For ``memory://bucket``, ``fs_path`` is ``/bucket`` (what fsspec's memory
    filesystem expects). For a plain local path the two values match.
    """
    uri = uri.rstrip("/")
    if "://" in uri:
        return uri, "/" + strip_scheme(uri)
    return uri, uri


def artifact_uri(bucket: str, *parts: str) -> str:
    """Build an artifact URI honoring DV_ARTIFACT_BACKEND.

    For backend=s3 / gcs / memory, returns a `<scheme>://bucket/parts...` URI.
    For backend=file (or unset), returns a relative `bucket/parts...` path
    suitable for the local filesystem.
    """
    scheme = _BACKEND_SCHEMES.get(
        os.environ.get("DV_ARTIFACT_BACKEND", "file").lower(), ""
    )
    return "/".join((f"{scheme}{bucket}", *parts)).rstrip("/")
