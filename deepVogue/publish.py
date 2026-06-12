"""Publish a trained checkpoint from Drive (or any source dir) to GCS/MinIO and update models.yaml."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from deepVogue.clients import get_artifact_fs, scheme_of, strip_scheme
from deepVogue.notifications import slack

log = logging.getLogger(__name__)

_SNAP_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")

# URI scheme → DV_ARTIFACT_BACKEND identifier.
_SCHEME_TO_BACKEND = {"gs": "gcs", "s3": "s3", "memory": "memory", "file": "file"}


def find_latest_snapshot(src_dir: Path) -> Path:
    candidates = [
        (int(m.group(1)), p)
        for p in Path(src_dir).iterdir()
        for m in [_SNAP_RE.search(p.name)]
        if m
    ]
    if not candidates:
        raise FileNotFoundError(f"no network-snapshot-*.pkl under {src_dir}")
    return max(candidates)[1]


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


def _align_backend_with_target(target_root: str) -> None:
    """Ensure DV_ARTIFACT_BACKEND matches the target URI scheme.

    Without this, a forgotten `export DV_ARTIFACT_BACKEND=gcs` silently routes
    writes to the local fs.
    """
    scheme = scheme_of(target_root)
    expected = _SCHEME_TO_BACKEND.get(scheme, "file")
    current = os.environ.get("DV_ARTIFACT_BACKEND", "file").lower()
    if current != expected:
        log.info(
            "auto-setting DV_ARTIFACT_BACKEND=%s to match target scheme %s",
            expected,
            scheme,
        )
        os.environ["DV_ARTIFACT_BACKEND"] = expected


def _copy_to_fs(src: Path, fs: Any, dst_path: str) -> None:
    with open(src, "rb") as r, fs.open(dst_path, "wb") as w:
        w.write(r.read())


def _upsert_registry_entry(fs: Any, yaml_path: str, entry: dict[str, Any]) -> None:
    """Read registry, replace any entry with same id, append new entry, write back.

    Note: not truly atomic (no if-generation-match); acceptable for v1.
    """
    if fs.exists(yaml_path):
        with fs.open(yaml_path, "r") as f:
            registry = yaml.safe_load(f) or []
    else:
        registry = []
    registry = [e for e in registry if e.get("id") != entry["id"]]
    registry.append(entry)
    with fs.open(yaml_path, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)


def _build_entry(
    *,
    model_id: str,
    backbone: str,
    pkl_uri: str,
    dataset_kind: str,
    default_trunc: float,
    factors: str | None,
    anchors_dir: str | None,
    fid: float | None,
) -> dict[str, Any]:
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
    return entry


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

    _align_backend_with_target(target_root)
    fs = get_artifact_fs()
    target_path = strip_scheme(target_root).rstrip("/")

    snapshot = find_latest_snapshot(Path(src_dir))
    if validate:
        _validate_pkl(snapshot)

    fid = _read_latest_fid(Path(src_dir), snapshot.name)
    pkl_uri = f"{target_root.rstrip('/')}/{model_id}/{snapshot.name}"
    _copy_to_fs(snapshot, fs, f"{target_path}/{model_id}/{snapshot.name}")

    entry = _build_entry(
        model_id=model_id,
        backbone=backbone,
        pkl_uri=pkl_uri,
        dataset_kind=dataset_kind,
        default_trunc=default_trunc,
        factors=factors,
        anchors_dir=anchors_dir,
        fid=fid,
    )
    _upsert_registry_entry(fs, f"{target_path}/models.yaml", entry)

    info = {"pkl": pkl_uri, "fid": fid, "model_id": model_id}
    slack.notify_success(
        "publish", f"published {model_id}", {"pkl": pkl_uri, "fid": str(fid)}
    )
    return info


def publish_dataset(
    *,
    name: str,
    src_dir: Path,
    target_root: str | None = None,
) -> dict[str, Any]:
    """
    Copy a prepared dataset from src_dir (Drive on Colab) to GCS so RunPod /
    Vertex jobs can train on it: ``<root>/<name>.zip`` becomes DV_DATASET_URI.
    ``frames_index.json`` rides along as ``<name>.frames_index.json`` when
    present (latent-cinema anchor map; not needed by the train job itself).
    """
    target_root = target_root or os.environ.get(
        "DV_DATASET_GCS_ROOT", "gs://deepvogue-datasets"
    )
    src_zip = Path(src_dir) / "dataset.zip"
    if not src_zip.exists():
        raise FileNotFoundError(f"no dataset.zip under {src_dir}")

    _align_backend_with_target(target_root)
    fs = get_artifact_fs()
    target_path = strip_scheme(target_root).rstrip("/")

    dataset_uri = f"{target_root.rstrip('/')}/{name}.zip"
    _copy_to_fs(src_zip, fs, f"{target_path}/{name}.zip")

    index_uri = None
    src_index = Path(src_dir) / "frames_index.json"
    if src_index.exists():
        index_uri = f"{target_root.rstrip('/')}/{name}.frames_index.json"
        _copy_to_fs(src_index, fs, f"{target_path}/{name}.frames_index.json")

    info = {"dataset_uri": dataset_uri, "frames_index_uri": index_uri, "name": name}
    slack.notify_success(
        "publish-dataset", f"uploaded dataset {name}", {"uri": dataset_uri}
    )
    log.info("dataset published: DV_DATASET_URI=%s", dataset_uri)
    return info


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m deepVogue.publish")
    sub = p.add_subparsers(dest="cmd")

    # default (no subcommand) = checkpoint publish, kept for `make publish` compat
    for target in (p, sub.add_parser("checkpoint")):
        target.add_argument("--model-id", required=target is not p)
        target.add_argument("--src-dir", type=Path, required=target is not p)
        target.add_argument("--backbone", default="sg3-t")
        target.add_argument("--dataset-kind", default="stills")
        target.add_argument("--default-trunc", default=0.7, type=float)
        target.add_argument("--factors", default=None)
        target.add_argument("--anchors-dir", default=None)
        target.add_argument("--validate", action="store_true")

    d = sub.add_parser("dataset", help="upload dataset.zip → GCS for RunPod/Vertex")
    d.add_argument("--name", required=True)
    d.add_argument("--src-dir", required=True, type=Path)
    d.add_argument("--dest", default=None, help="default $DV_DATASET_GCS_ROOT or gs://deepvogue-datasets")

    args = p.parse_args()
    if args.cmd == "dataset":
        info = publish_dataset(name=args.name, src_dir=args.src_dir, target_root=args.dest)
    else:
        if not args.model_id or not args.src_dir:
            p.error("--model-id and --src-dir are required")
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
