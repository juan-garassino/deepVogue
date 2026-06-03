import os
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from deepVogue.orchestration.flows import (
    prepare_flow,
    train_flow,
    publish_flow,
    walk_flow,
    pipeline_stills,
)


def test_prepare_flow_returns_dataset_uri(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(src / f"{i}.png")
    out = prepare_flow(
        source_uri=str(src),
        dataset_name="t_nano",
        res=64,
        kind="stills",
        target_uri="memory://deepvogue-datasets",
        backend="local",
    )
    assert out["dataset_uri"].endswith(".zip")


def test_train_flow_writes_fake_pkl(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    out = train_flow(
        dataset_name="t_nano",
        cfg="stylegan3-t",
        kimg=50,
        gamma=2.0,
        batch=8,
        res=64,
        target_uri="memory://deepvogue-models/t_nano",
        backend="local",
    )
    assert out["kimg"] == 50
    assert out["fid"] < 200.0


def test_pipeline_stills_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(src / f"{i}.png")
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        result = pipeline_stills(
            source_uri=str(src),
            dataset_name="t_nano",
            model_id="t_nano_v1",
            res=64,
            kimg=50,
            gamma=2.0,
            batch=8,
            walk_steps=10,
            walk_fps=12,
            backend="local",
        )
    assert "walk_uri" in result
    assert "pkl" in result["train"]
