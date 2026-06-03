"""MLflow logging helpers for deepVogue training runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow

log = logging.getLogger(__name__)


def _read_fid_jsonl(run_dir: Path) -> list[dict]:
    jsonl = run_dir / "metric-fid50k_full.jsonl"
    if not jsonl.exists():
        return []
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed FID jsonl line: %s", line[:80])
    return rows


def _snapshot_step(name: str) -> int:
    # e.g. network-snapshot-005200.pkl -> 5200
    try:
        return int(name.split("network-snapshot-")[1].split(".")[0])
    except (IndexError, ValueError):
        return 0


def log_training_run(
    run_dir: Path,
    *,
    dataset_name: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int,
    extra_tags: dict[str, str] | None = None,
    log_snapshot_every: int = 1,
) -> str:
    """Log a deepVogue training run dir to MLflow. Returns the MLflow run_id."""
    run_dir = Path(run_dir)
    mlflow.set_experiment(dataset_name)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "cfg": cfg,
                "kimg": kimg,
                "gamma": gamma,
                "batch": batch,
                "res": res,
                "run_dir": str(run_dir),
            }
        )
        if extra_tags:
            mlflow.set_tags(extra_tags)
        fid_rows = _read_fid_jsonl(run_dir)
        snapshots_logged = 0
        for row in fid_rows:
            snap = row.get("snapshot_pkl", "")
            step = _snapshot_step(snap)
            fid = row.get("results", {}).get("fid50k_full")
            if fid is not None:
                mlflow.log_metric("fid50k_full", float(fid), step=step)
            pkl = run_dir / snap
            if pkl.exists() and snapshots_logged % log_snapshot_every == 0:
                mlflow.log_artifact(str(pkl), artifact_path="snapshots")
            snapshots_logged += 1
        return run.info.run_id
