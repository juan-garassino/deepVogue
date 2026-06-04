import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml
from deepVogue.publish import publish_checkpoint, find_latest_snapshot


def _make_drive_snapshot(tmp_path: Path) -> Path:
    drive = tmp_path / "drive_sync" / "tarot"
    drive.mkdir(parents=True)
    (drive / "network-snapshot-000200.pkl").write_bytes(b"\x00" * 16)
    (drive / "network-snapshot-000400.pkl").write_bytes(b"\x00" * 16)
    (drive / "metric-fid50k_full.jsonl").write_text(
        json.dumps(
            {
                "snapshot_pkl": "network-snapshot-000400.pkl",
                "results": {"fid50k_full": 98.6},
            }
        )
        + "\n"
    )
    return drive


def test_find_latest_snapshot(tmp_path):
    drive = _make_drive_snapshot(tmp_path)
    latest = find_latest_snapshot(drive)
    assert latest.name == "network-snapshot-000400.pkl"


def test_publish_uploads_and_appends_models_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    drive = _make_drive_snapshot(tmp_path)

    # bypass real legacy.load_network_pkl
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        info = publish_checkpoint(
            model_id="tarot_v1",
            src_dir=drive,
            backbone="sg3-t",
            dataset_kind="stills",
            default_trunc=0.7,
        )

    import fsspec

    fs = fsspec.filesystem("memory")
    files = fs.ls("/deepvogue-models")
    files_str = [f if isinstance(f, str) else f["name"] for f in files]
    assert any("tarot_v1" in f for f in files_str)
    assert any("models.yaml" in f for f in files_str)

    with fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 1
    assert registry[0]["id"] == "tarot_v1"
    assert registry[0]["pkl"].endswith("network-snapshot-000400.pkl")
    assert registry[0]["dataset_kind"] == "stills"
    assert info["fid"] == 98.6


def test_publish_auto_derives_backend_from_target(tmp_path, monkeypatch):
    monkeypatch.delenv("DV_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    drive = _make_drive_snapshot(tmp_path)
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        publish_checkpoint(
            model_id="t", src_dir=drive, backbone="sg3-t", dataset_kind="stills"
        )
    assert os.environ["DV_ARTIFACT_BACKEND"] == "memory"


def test_publish_appends_when_registry_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    drive = _make_drive_snapshot(tmp_path)

    import fsspec

    fs = fsspec.filesystem("memory")
    with fs.open("/deepvogue-models/models.yaml", "w") as f:
        yaml.safe_dump(
            [{"id": "existing", "backbone": "sg3-t", "pkl": "memory://existing.pkl"}], f
        )

    with patch("deepVogue.publish._validate_pkl", return_value=None):
        publish_checkpoint(
            model_id="tarot_v1", src_dir=drive, backbone="sg3-t", dataset_kind="stills"
        )

    with fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 2
    assert {e["id"] for e in registry} == {"existing", "tarot_v1"}
