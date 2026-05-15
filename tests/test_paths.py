import os
import pytest
from deepVogue import _paths


def test_resolve_defaults_to_tmp(monkeypatch):
    for k in ("DV_DATA_DIR", "DV_DATASET_DIR", "DV_RUN_DIR", "DV_DRIVE_SYNC",
              "DV_NETWORK_PKL", "DV_ANCHORS_DIR", "DV_WALKS_DIR", "DV_DATASET_NAME"):
        monkeypatch.delenv(k, raising=False)
    p = _paths.resolve()
    assert str(p.run_dir).startswith("/tmp/")
    assert p.drive_sync is None
    assert p.dataset_name is None


def test_required_env_raises(monkeypatch):
    monkeypatch.delenv("DV_NETWORK_PKL", raising=False)
    with pytest.raises(RuntimeError, match="DV_NETWORK_PKL"):
        _paths.resolve(require=("network_pkl",))


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DV_DRIVE_SYNC", str(tmp_path / "sync"))
    monkeypatch.delenv("DV_DATASET_NAME", raising=False)
    p = _paths.resolve()
    assert p.data_dir == tmp_path / "data"
    assert p.drive_sync == tmp_path / "sync"


def test_dataset_name_scopes_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("DV_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("DV_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DV_DRIVE_SYNC", str(tmp_path / "drive"))
    monkeypatch.setenv("DV_ANCHORS_DIR", str(tmp_path / "anchors"))
    monkeypatch.setenv("DV_WALKS_DIR", str(tmp_path / "walks"))
    monkeypatch.setenv("DV_DATASET_NAME", "tarot")
    p = _paths.resolve()
    assert p.dataset_name == "tarot"
    assert p.dataset_dir == tmp_path / "datasets" / "tarot"
    assert p.run_dir == tmp_path / "runs" / "tarot"
    assert p.drive_sync == tmp_path / "drive" / "tarot"
    assert p.anchors_dir == tmp_path / "anchors" / "tarot"
    assert p.walks_dir == tmp_path / "walks" / "tarot"
    # data_dir is NOT scoped (raw input is per-domain manually)
    assert p.data_dir == p.data_dir


def test_latest_snapshot_picks_highest_kimg(monkeypatch, tmp_path):
    runs = tmp_path / "runs" / "tarot"
    r1 = runs / "00000-tarot"; r1.mkdir(parents=True)
    r2 = runs / "00001-tarot"; r2.mkdir(parents=True)
    (r1 / "network-snapshot-000100.pkl").write_text("a")
    (r2 / "network-snapshot-000200.pkl").write_text("b")
    (r2 / "network-snapshot-000150.pkl").write_text("c")
    monkeypatch.setenv("DV_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("DV_DRIVE_SYNC", raising=False)
    monkeypatch.setenv("DV_DATASET_NAME", "tarot")
    latest = _paths.latest_snapshot()
    assert latest is not None and latest.name == "network-snapshot-000200.pkl"


def test_list_runs_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("DV_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("DV_DRIVE_SYNC", raising=False)
    monkeypatch.setenv("DV_DATASET_NAME", "nope")
    assert _paths.list_runs() == []
