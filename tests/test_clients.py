import os
import pytest
from deepVogue.clients import artifact_uri, get_artifact_fs, resolve_uri


def test_s3_backend_with_minio_endpoint(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("DV_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "miniopass")
    fs = get_artifact_fs()
    assert fs.protocol == ("s3", "s3a")
    assert fs.client_kwargs["endpoint_url"] == "http://localhost:9000"


def test_gcs_backend(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "gcs")
    monkeypatch.delenv("DV_S3_ENDPOINT_URL", raising=False)
    fs = get_artifact_fs()
    assert "gcs" in (
        fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
    ) or "gs" in (fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,))


def test_memory_backend_for_tests(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    fs = get_artifact_fs()
    assert fs.protocol == "memory"


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "wat")
    with pytest.raises(ValueError, match="unknown"):
        get_artifact_fs()


def test_resolve_uri_passthrough_absolute():
    assert resolve_uri("s3://b/k") == "s3://b/k"
    assert resolve_uri("gs://b/k") == "gs://b/k"


def test_resolve_uri_resolves_relative_to_models_root(monkeypatch):
    monkeypatch.setenv("DV_MODELS_ROOT", "/data/runs")
    assert resolve_uri("tarot/snap.pkl") == "/data/runs/tarot/snap.pkl"


def test_artifact_uri_s3_backend(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "s3")
    assert artifact_uri("deepvogue-datasets") == "s3://deepvogue-datasets"
    assert (
        artifact_uri("deepvogue-datasets", "tarot", "v1")
        == "s3://deepvogue-datasets/tarot/v1"
    )


def test_artifact_uri_gcs_backend(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "gcs")
    assert artifact_uri("deepvogue-models") == "gs://deepvogue-models"
    assert (
        artifact_uri("deepvogue-models", "tarot_v1") == "gs://deepvogue-models/tarot_v1"
    )


def test_artifact_uri_memory_backend(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    assert artifact_uri("deepvogue-walks") == "memory://deepvogue-walks"
    assert (
        artifact_uri("deepvogue-walks", "abc.mp4") == "memory://deepvogue-walks/abc.mp4"
    )


def test_artifact_uri_file_backend_default(monkeypatch):
    monkeypatch.delenv("DV_ARTIFACT_BACKEND", raising=False)
    assert artifact_uri("deepvogue-datasets") == "deepvogue-datasets"
    assert artifact_uri("deepvogue-datasets", "tarot") == "deepvogue-datasets/tarot"
