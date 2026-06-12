"""Integration test: runs against an already-running docker-compose nano stack.

Skipped unless DV_NANO_SMOKE=1. Triggered by `make nano-smoke` after `make nano-up`.
"""

import os
import time
from pathlib import Path

import pytest
import requests

NANO = pytest.mark.skipif(
    os.environ.get("DV_NANO_SMOKE") != "1",
    reason="set DV_NANO_SMOKE=1 + `make nano-up` first",
)


@NANO
def test_mlflow_reachable():
    r = requests.get("http://localhost:5000", timeout=5)
    assert r.status_code == 200


@NANO
def test_prefect_reachable():
    r = requests.get("http://localhost:4200/api/health", timeout=5)
    assert r.status_code == 200


@NANO
def test_fastapi_reachable():
    r = requests.get("http://localhost:8080/health", timeout=5)
    assert r.status_code == 200


@NANO
def test_pipeline_stills_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("DV_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "miniopass")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    from PIL import Image

    src = tmp_path / "raw"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 80, 0, 0)).save(src / f"{i}.png")

    from deepVogue.orchestration.flows import pipeline_stills

    out = pipeline_stills(
        source_uri=str(src),
        dataset_name="smoke_stills",
        model_id="smoke_stills_v1",
        res=64,
        kimg=50,
        gamma=2.0,
        batch=8,
        walk_steps=10,
        walk_fps=12,
        backend="local",
    )
    assert "walk_uri" in out
    # verify mp4 actually exists in MinIO
    import fsspec

    fs = fsspec.filesystem(
        "s3", client_kwargs={"endpoint_url": "http://localhost:9000"}
    )
    path = out["walk_uri"].split("://", 1)[1]
    assert fs.exists(path), f"mp4 not in MinIO at {path}"
