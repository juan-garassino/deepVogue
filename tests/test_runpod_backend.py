"""Tests for the RunPod orchestration backend.

The real ``runpod`` SDK is replaced with a stub module so tests don't need
network or API keys.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_runpod(monkeypatch):
    """Install a stub `runpod` module that records calls and returns canned data."""
    mod = types.ModuleType("runpod")
    mod.api_key = None
    mod.create_pod = MagicMock(return_value={"id": "pod-abc123"})
    mod.get_pod = MagicMock(
        return_value={"id": "pod-abc123", "desiredStatus": "TERMINATED"}
    )
    mod.stop_pod = MagicMock(return_value=None)
    mod.terminate_pod = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "runpod", mod)
    return mod


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test-key")
    monkeypatch.setenv("RUNPOD_IMAGE", "ghcr.io/org/deepvogue-train:latest")
    monkeypatch.setenv("DV_DATASET_URI", "gs://deepvogue-datasets/tarot.zip")
    monkeypatch.setenv("DV_RUN_URI", "gs://deepvogue-runs/tarot/2026-06-06")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "gs://deepvogue-models")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON", '{"type":"service_account"}'
    )
    return monkeypatch


def _fast_poll(monkeypatch):
    monkeypatch.setattr(
        "deepVogue.orchestration.backends.runpod.POLL_INTERVAL_SECONDS", 0
    )


def test_train_submits_pod_with_expected_env(fake_runpod, env, monkeypatch):
    _fast_poll(monkeypatch)
    from deepVogue.orchestration.backends import runpod as backend

    out = backend.train(
        dataset_name="tarot",
        cfg="stylegan3-t",
        kimg=200,
        gamma=2.0,
        batch=32,
        res=256,
    )

    assert fake_runpod.create_pod.call_count == 1
    kwargs = fake_runpod.create_pod.call_args.kwargs
    assert kwargs["image_name"] == "ghcr.io/org/deepvogue-train:latest"
    assert kwargs["gpu_count"] == 1
    env_passed = kwargs["env"]
    assert env_passed["DV_DATASET_URI"] == "gs://deepvogue-datasets/tarot.zip"
    assert env_passed["DV_RUN_URI"] == "gs://deepvogue-runs/tarot/2026-06-06"
    assert env_passed["DV_MODEL_ID"] == "tarot"
    assert env_passed["DV_CFG"] == "stylegan3-t"
    assert env_passed["DV_KIMG"] == "200"
    assert env_passed["DV_GAMMA"] == "2.0"
    assert env_passed["DV_BATCH"] == "32"
    assert env_passed["DV_RES"] == "256"
    assert env_passed["MLFLOW_TRACKING_URI"] == "https://mlflow.example.com"
    assert env_passed["DV_PUBLISH_TARGET"] == "gs://deepvogue-models"
    # The pod needs both the SA JSON and an API key to self-terminate.
    assert env_passed["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    assert env_passed["RUNPOD_API_KEY"] == "rp-test-key"

    assert out["pod_id"] == "pod-abc123"
    assert out["kimg"] == 200
    assert out["pkl"].endswith("network-snapshot-000200.pkl")
    assert out["pkl"].startswith("gs://deepvogue-runs/tarot/2026-06-06")
    assert "elapsed_s" in out

    # Guard terminate is best-effort; runs regardless of lifecycle path.
    fake_runpod.terminate_pod.assert_called_once_with("pod-abc123")


def test_train_treats_gone_pod_as_success(fake_runpod, env, monkeypatch):
    _fast_poll(monkeypatch)
    fake_runpod.get_pod.return_value = None
    from deepVogue.orchestration.backends import runpod as backend

    out = backend.train(
        dataset_name="tarot",
        cfg="stylegan3-t",
        kimg=10,
        gamma=2.0,
        batch=32,
        res=64,
    )
    assert out["pod_id"] == "pod-abc123"


def test_train_raises_when_pod_fails(fake_runpod, env, monkeypatch):
    _fast_poll(monkeypatch)
    fake_runpod.get_pod.return_value = {"id": "pod-abc123", "desiredStatus": "FAILED"}
    from deepVogue.orchestration.backends import runpod as backend

    with pytest.raises(RuntimeError, match="FAILED"):
        backend.train(
            dataset_name="tarot",
            cfg="stylegan3-t",
            kimg=10,
            gamma=2.0,
            batch=32,
            res=64,
        )
    # Even on failure, we still try to terminate to stop billing.
    fake_runpod.terminate_pod.assert_called_once_with("pod-abc123")


def test_train_timeout_raises_and_terminates(fake_runpod, env, monkeypatch):
    _fast_poll(monkeypatch)
    monkeypatch.setenv("RUNPOD_MAX_TRAIN_HOURS", "0")  # immediate timeout
    fake_runpod.get_pod.return_value = {"id": "pod-abc123", "desiredStatus": "RUNNING"}
    from deepVogue.orchestration.backends import runpod as backend

    with pytest.raises(RuntimeError, match="TIMEOUT"):
        backend.train(
            dataset_name="tarot",
            cfg="stylegan3-t",
            kimg=10,
            gamma=2.0,
            batch=32,
            res=64,
        )
    fake_runpod.terminate_pod.assert_called_once_with("pod-abc123")


def test_train_resume_passes_resume_uri(fake_runpod, env, monkeypatch):
    _fast_poll(monkeypatch)
    from deepVogue.orchestration.backends import runpod as backend

    backend.train(
        dataset_name="tarot",
        cfg="stylegan3-t",
        kimg=100,
        gamma=2.0,
        batch=32,
        res=128,
        resume_from="gs://deepvogue-runs/tarot/prev/network-snapshot-005000.pkl",
    )
    env_passed = fake_runpod.create_pod.call_args.kwargs["env"]
    assert (
        env_passed["DV_RESUME_FROM"]
        == "gs://deepvogue-runs/tarot/prev/network-snapshot-005000.pkl"
    )


def test_train_requires_api_key(fake_runpod, env, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    from deepVogue.orchestration.backends import runpod as backend

    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        backend.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


def test_train_requires_dataset_uri(fake_runpod, env, monkeypatch):
    monkeypatch.delenv("DV_DATASET_URI", raising=False)
    from deepVogue.orchestration.backends import runpod as backend

    with pytest.raises(RuntimeError, match="DV_DATASET_URI"):
        backend.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


def test_train_requires_sa_json(fake_runpod, env, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    from deepVogue.orchestration.backends import runpod as backend

    with pytest.raises(RuntimeError, match="GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        backend.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


def test_non_train_ops_delegate_to_local(fake_runpod, env, monkeypatch, tmp_path):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    from PIL import Image

    src = tmp_path / "stills"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (32, 32), (i * 100, 0, 0)).save(src / f"{i}.png")

    from deepVogue.orchestration.backends import runpod as backend

    out = backend.prepare(
        source_uri=str(src),
        dataset_name="smoke_nano",
        res=32,
        kind="stills",
        target_uri="memory://deepvogue-datasets-runpod",
    )
    assert out["dataset_uri"].endswith(".zip")
