import pytest
from deepVogue.orchestration.backends import get_backend, BackendOp


def test_get_local_backend_has_required_ops():
    b = get_backend("local")
    for op in ("prepare", "train", "publish", "project", "walk", "eval"):
        assert hasattr(b, op), f"local backend missing op: {op}"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_colab_backend_raises_notimplemented_in_v1():
    b = get_backend("colab")
    with pytest.raises(NotImplementedError):
        b.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


def test_runpod_backend_raises_notimplemented_in_v1():
    b = get_backend("runpod")
    with pytest.raises(NotImplementedError):
        b.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


import os
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from deepVogue.orchestration.backends import get_backend


def test_local_prepare_creates_dataset_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    src = tmp_path / "stills"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 80, 0, 0)).save(src / f"{i}.png")
    b = get_backend("local")
    out = b.prepare(
        source_uri=str(src),
        dataset_name="tarot_nano",
        res=64,
        kind="stills",
        target_uri="memory://deepvogue-datasets",
    )
    assert out["dataset_uri"].endswith(".zip")
    import fsspec

    fs = fsspec.filesystem("memory")
    assert fs.exists(out["dataset_uri"].replace("memory://", "/"))


def test_local_publish_delegates_to_publish_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "network-snapshot-000200.pkl").write_bytes(b"\x00")
    b = get_backend("local")
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        info = b.publish(
            model_id="tarot_v1",
            src_dir=str(drive),
            backbone="sg3-t",
            dataset_kind="stills",
        )
    assert info["model_id"] == "tarot_v1"
