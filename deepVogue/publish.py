"""Publish a trained checkpoint from Drive (or any source dir) to GCS/MinIO and update models.yaml."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from deepVogue.clients import get_artifact_fs
from deepVogue.notifications import slack

log = logging.getLogger(__name__)

_SNAP_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def find_latest_snapshot(src_dir: Path) -> Path:
    candidates = []
    for p in Path(src_dir).iterdir():
        m = _SNAP_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"no network-snapshot-*.pkl under {src_dir}")
    candidates.sort()
    return candidates[-1][1]


def _read_latest_fid(src_dir: Path, snapshot_name: str) -> float | None:
    jsonl = Path(src_dir) / "metric-fid50k_full.jsonl"
    if not jsonl.exists():
        return None
    fid = None
    for line in jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("snapshot_pkl") == snapshot_name:
            fid = row.get("results", {}).get("fid50k_full")
    return float(fid) if fid is not None else None


def _validate_pkl(pkl_path: Path) -> None:
    """Hook for runtime pkl validation. Overridden in tests."""
    from deepVogue import legacy  # noqa: F401

    with open(pkl_path, "rb") as f:
        legacy.load_network_pkl(f)  # type: ignore[attr-defined]


def _strip_proto(uri: str) -> str:
    return uri.split("://", 1)[1] if "://" in uri else uri


def publish_checkpoint(
    *,
    model_id: str,
    src_dir: Path,
    backbone: str = "sg3-t",
    dataset_kind: str = "stills",
    default_trunc: float = 0.7,
    factors: str | None = None,
    anchors_dir: str | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """
    Copy latest snapshot from src_dir → DV_PUBLISH_TARGET; append/update models.yaml.
    Returns info dict (pkl uri, fid, etc.). Posts to Slack on success.
    """
    target_root = os.environ.get("DV_PUBLISH_TARGET")
    if not target_root:
        raise RuntimeError("DV_PUBLISH_TARGET must be set (e.g. gs://deepvogue-models)")
    fs = get_artifact_fs()
    target_path = _strip_proto(target_root).rstrip("/")

    snapshot = find_latest_snapshot(Path(src_dir))
    if validate:
        _validate_pkl(snapshot)

    fid = _read_latest_fid(Path(src_dir), snapshot.name)
    target_pkl_path = f"{target_path}/{model_id}/{snapshot.name}"
    pkl_uri = f"{target_root.rstrip('/')}/{model_id}/{snapshot.name}"

    with open(snapshot, "rb") as src, fs.open(target_pkl_path, "wb") as dst:
        dst.write(src.read())

    yaml_path = f"{target_path}/models.yaml"
    if fs.exists(yaml_path):
        with fs.open(yaml_path, "r") as f:
            registry = yaml.safe_load(f) or []
    else:
        registry = []

    entry: dict[str, Any] = {
        "id": model_id,
        "backbone": backbone,
        "pkl": pkl_uri,
        "dataset_kind": dataset_kind,
        "default_trunc": default_trunc,
    }
    if factors:
        entry["factors"] = factors
    if anchors_dir:
        entry["anchors_dir"] = anchors_dir
    if fid is not None:
        entry["fid"] = fid

    registry = [e for e in registry if e.get("id") != model_id]
    registry.append(entry)

    with fs.open(yaml_path, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    info = {"pkl": pkl_uri, "fid": fid, "model_id": model_id}
    slack.notify_success(
        "publish", f"published {model_id}", {"pkl": pkl_uri, "fid": str(fid)}
    )
    return info


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m deepVogue.publish")
    p.add_argument("--model-id", required=True)
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--backbone", default="sg3-t")
    p.add_argument("--dataset-kind", default="stills")
    p.add_argument("--default-trunc", default=0.7, type=float)
    p.add_argument("--factors", default=None)
    p.add_argument("--anchors-dir", default=None)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    info = publish_checkpoint(
        model_id=args.model_id,
        src_dir=args.src_dir,
        backbone=args.backbone,
        dataset_kind=args.dataset_kind,
        default_trunc=args.default_trunc,
        factors=args.factors,
        anchors_dir=args.anchors_dir,
        validate=args.validate,
    )
    print(info)


if __name__ == "__main__":
    _main()
