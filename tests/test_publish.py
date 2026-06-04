import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml
from deepVogue.publish import publish_checkpoint, find_latest_snapshot


@pytest.fixture
def drive_snapshot(tmp_path: Path) -> Path:
    """Drive-style snapshot dir with two pkls + a FID metric jsonl."""
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


@pytest.fixture
def memory_fs():
    import fsspec

    fs = fsspec.filesystem("memory")
    # Clear any state left over from previous tests.
    for p in list(fs.ls("/", detail=False)):
        try:
            fs.rm(p, recursive=True)
        except (FileNotFoundError, IsADirectoryError):
            pass
    return fs


@pytest.fixture
def publish_env(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)


@pytest.fixture
def stub_validate():
    with patch("deepVogue.publish._validate_pkl", return_value=None) as m:
        yield m


def test_find_latest_snapshot(drive_snapshot):
    assert find_latest_snapshot(drive_snapshot).name == "network-snapshot-000400.pkl"


def test_publish_uploads_and_appends_models_yaml(
    drive_snapshot, memory_fs, publish_env, stub_validate
):
    info = publish_checkpoint(
        model_id="tarot_v1",
        src_dir=drive_snapshot,
        backbone="sg3-t",
        dataset_kind="stills",
        default_trunc=0.7,
    )

    files = memory_fs.ls("/deepvogue-models")
    files_str = [f if isinstance(f, str) else f["name"] for f in files]
    assert any("tarot_v1" in f for f in files_str)
    assert any("models.yaml" in f for f in files_str)

    with memory_fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 1
    assert registry[0]["id"] == "tarot_v1"
    assert registry[0]["pkl"].endswith("network-snapshot-000400.pkl")
    assert registry[0]["dataset_kind"] == "stills"
    assert info["fid"] == 98.6


def test_publish_auto_derives_backend_from_target(
    drive_snapshot, memory_fs, monkeypatch, stub_validate
):
    monkeypatch.delenv("DV_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    publish_checkpoint(
        model_id="t", src_dir=drive_snapshot, backbone="sg3-t", dataset_kind="stills"
    )
    assert os.environ["DV_ARTIFACT_BACKEND"] == "memory"


def test_publish_appends_when_registry_exists(
    drive_snapshot, memory_fs, publish_env, stub_validate
):
    with memory_fs.open("/deepvogue-models/models.yaml", "w") as f:
        yaml.safe_dump(
            [{"id": "existing", "backbone": "sg3-t", "pkl": "memory://existing.pkl"}], f
        )

    publish_checkpoint(
        model_id="tarot_v1",
        src_dir=drive_snapshot,
        backbone="sg3-t",
        dataset_kind="stills",
    )

    with memory_fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 2
    assert {e["id"] for e in registry} == {"existing", "tarot_v1"}
