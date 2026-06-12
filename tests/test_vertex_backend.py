"""Tests for the Vertex AI orchestration backend.

The real ``google.cloud.aiplatform`` SDK is replaced with a stub module so
tests don't need network or GCP credentials.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


class _FakeState:
    def __init__(self, name: str):
        self.name = name


class _FakeJob:
    def __init__(self, fail: bool = False):
        self.resource_name = "projects/p/locations/europe-west1/customJobs/123"
        self.state = _FakeState("JOB_STATE_SUCCEEDED")
        self._fail = fail
        self.submit = MagicMock()
        if fail:
            self.state = _FakeState("JOB_STATE_FAILED")
            self.wait = MagicMock(side_effect=RuntimeError("job failed"))
        else:
            self.wait = MagicMock()


@pytest.fixture
def fake_aiplatform(monkeypatch):
    """Install a stub `google.cloud.aiplatform` that records calls."""
    aiplatform = types.ModuleType("google.cloud.aiplatform")
    aiplatform.init = MagicMock()
    aiplatform._job = _FakeJob()
    aiplatform.CustomJob = MagicMock(return_value=aiplatform._job)

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.aiplatform = aiplatform
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", aiplatform)
    return aiplatform


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT", "garassino-ml")
    monkeypatch.setenv(
        "VERTEX_IMAGE",
        "europe-west1-docker.pkg.dev/garassino-ml/deepvogue/deepvogue-train:latest",
    )
    monkeypatch.setenv("DV_DATASET_URI", "gs://deepvogue-datasets/tarot.zip")
    monkeypatch.setenv("DV_RUN_URI", "gs://deepvogue-runs/tarot/2026-06-12")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "gs://deepvogue-models")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    return monkeypatch


def _worker_pool(fake_aiplatform) -> dict:
    return fake_aiplatform.CustomJob.call_args.kwargs["worker_pool_specs"][0]


def test_train_submits_job_with_expected_spec(fake_aiplatform, env):
    from deepVogue.orchestration.backends import vertex as backend

    out = backend.train(
        dataset_name="tarot",
        cfg="stylegan3-t",
        kimg=200,
        gamma=2.0,
        batch=32,
        res=256,
    )

    fake_aiplatform.init.assert_called_once_with(
        project="garassino-ml", location="europe-west1"
    )
    pool = _worker_pool(fake_aiplatform)
    assert pool["machine_spec"]["machine_type"] == "g2-standard-8"
    assert pool["machine_spec"]["accelerator_type"] == "NVIDIA_L4"
    assert pool["machine_spec"]["accelerator_count"] == 1
    assert pool["container_spec"]["image_uri"].endswith("deepvogue-train:latest")

    env_passed = {e["name"]: e["value"] for e in pool["container_spec"]["env"]}
    assert env_passed["DV_DATASET_URI"] == "gs://deepvogue-datasets/tarot.zip"
    assert env_passed["DV_RUN_URI"] == "gs://deepvogue-runs/tarot/2026-06-12"
    assert env_passed["DV_MODEL_ID"] == "tarot"
    assert env_passed["DV_KIMG"] == "200"
    assert env_passed["MLFLOW_TRACKING_URI"] == "https://mlflow.example.com"
    assert env_passed["DV_PUBLISH_TARGET"] == "gs://deepvogue-models"
    # No SA key and no RunPod plumbing on Vertex — ambient ADC + platform teardown.
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in env_passed
    assert "RUNPOD_API_KEY" not in env_passed

    submit_kwargs = fake_aiplatform._job.submit.call_args.kwargs
    assert (
        submit_kwargs["service_account"]
        == "deepvogue-trainer-sa@garassino-ml.iam.gserviceaccount.com"
    )
    assert submit_kwargs["timeout"] == 24 * 3600

    assert out["job"].endswith("customJobs/123")
    assert out["kimg"] == 200
    assert (
        out["pkl"] == "gs://deepvogue-runs/tarot/2026-06-12/network-snapshot-000200.pkl"
    )
    assert "elapsed_s" in out


def test_train_falls_back_to_runpod_image(fake_aiplatform, env, monkeypatch):
    monkeypatch.delenv("VERTEX_IMAGE")
    monkeypatch.setenv("RUNPOD_IMAGE", "ghcr.io/org/deepvogue-train:latest")
    from deepVogue.orchestration.backends import vertex as backend

    backend.train(
        dataset_name="tarot", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
    )
    pool = _worker_pool(fake_aiplatform)
    assert pool["container_spec"]["image_uri"] == "ghcr.io/org/deepvogue-train:latest"


def test_train_machine_overrides(fake_aiplatform, env, monkeypatch):
    monkeypatch.setenv("VERTEX_MACHINE_TYPE", "n1-standard-8")
    monkeypatch.setenv("VERTEX_ACCELERATOR", "NVIDIA_TESLA_T4")
    from deepVogue.orchestration.backends import vertex as backend

    backend.train(
        dataset_name="tarot", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
    )
    pool = _worker_pool(fake_aiplatform)
    assert pool["machine_spec"]["machine_type"] == "n1-standard-8"
    assert pool["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"


def test_train_raises_when_job_fails(fake_aiplatform, env):
    fake_aiplatform._job = _FakeJob(fail=True)
    fake_aiplatform.CustomJob = MagicMock(return_value=fake_aiplatform._job)
    from deepVogue.orchestration.backends import vertex as backend

    with pytest.raises(RuntimeError, match="JOB_STATE_FAILED"):
        backend.train(
            dataset_name="tarot",
            cfg="stylegan3-t",
            kimg=10,
            gamma=2.0,
            batch=32,
            res=64,
        )


def test_train_requires_gcp_project(fake_aiplatform, env, monkeypatch):
    monkeypatch.delenv("GCP_PROJECT")
    from deepVogue.orchestration.backends import vertex as backend

    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        backend.train(
            dataset_name="tarot",
            cfg="stylegan3-t",
            kimg=10,
            gamma=2.0,
            batch=32,
            res=64,
        )


def test_get_backend_resolves_vertex():
    from deepVogue.orchestration.backends import get_backend, vertex

    assert get_backend("vertex") is vertex


def test_non_gpu_ops_delegate_to_local(monkeypatch):
    from deepVogue.orchestration.backends import vertex as backend

    called = {}
    monkeypatch.setattr(
        backend, "_delegate", lambda op, **kw: called.setdefault(op, kw)
    )
    backend.prepare(x=1)
    backend.publish(y=2)
    assert called == {"prepare": {"x": 1}, "publish": {"y": 2}}


def test_fake_train_not_forwarded_by_default(fake_aiplatform, env, monkeypatch):
    monkeypatch.delenv("DV_FAKE_TRAIN", raising=False)
    from deepVogue.orchestration.backends import vertex as backend

    backend.train(
        dataset_name="tarot", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
    )
    env_passed = {
        e["name"] for e in _worker_pool(fake_aiplatform)["container_spec"]["env"]
    }
    assert "DV_FAKE_TRAIN" not in env_passed


def test_fake_train_opt_in_forwarded(fake_aiplatform, env, monkeypatch):
    monkeypatch.setenv("DV_FAKE_TRAIN", "1")
    from deepVogue.orchestration.backends import vertex as backend

    backend.train(
        dataset_name="tarot", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
    )
    env_passed = {
        e["name"]: e["value"]
        for e in _worker_pool(fake_aiplatform)["container_spec"]["env"]
    }
    assert env_passed["DV_FAKE_TRAIN"] == "1"
