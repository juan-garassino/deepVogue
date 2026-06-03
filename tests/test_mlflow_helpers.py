import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from deepVogue.tracking.mlflow_helpers import log_training_run


def _make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "00000-tarot-stylegan3-t-gpus1-batch32-gamma2"
    run.mkdir()
    # FID jsonl
    fid = run / "metric-fid50k_full.jsonl"
    fid.write_text(
        json.dumps({"snapshot_pkl": "network-snapshot-000200.pkl", "results": {"fid50k_full": 142.3}}) + "\n" +
        json.dumps({"snapshot_pkl": "network-snapshot-000400.pkl", "results": {"fid50k_full": 98.6}}) + "\n"
    )
    (run / "network-snapshot-000200.pkl").write_bytes(b"\x00" * 16)
    (run / "network-snapshot-000400.pkl").write_bytes(b"\x00" * 16)
    return run


def test_log_training_run_creates_experiment_and_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "memory://")
    run_dir = _make_run_dir(tmp_path)

    with patch("deepVogue.tracking.mlflow_helpers.mlflow") as mf:
        mf.set_experiment.return_value = None
        mf.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r1"))
        mf.start_run.return_value.__exit__.return_value = False
        log_training_run(run_dir, dataset_name="tarot", cfg="stylegan3-t", kimg=5000, gamma=2.0, batch=32, res=512)

        mf.set_experiment.assert_called_once_with("tarot")
        mf.log_params.assert_called_once()
        params = mf.log_params.call_args.args[0]
        assert params["cfg"] == "stylegan3-t"
        assert params["gamma"] == 2.0
        # metrics: two FID steps
        assert mf.log_metric.call_count >= 2
        metric_calls = [c.args for c in mf.log_metric.call_args_list]
        assert any(c[0] == "fid50k_full" and c[1] == 142.3 for c in metric_calls)
        assert any(c[0] == "fid50k_full" and c[1] == 98.6 for c in metric_calls)


def test_log_training_run_skips_when_no_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "memory://")
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with patch("deepVogue.tracking.mlflow_helpers.mlflow") as mf:
        mf.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r1"))
        mf.start_run.return_value.__exit__.return_value = False
        log_training_run(run_dir, dataset_name="tarot", cfg="stylegan3-t", kimg=0, gamma=2.0, batch=32, res=512)
        mf.log_metric.assert_not_called()
