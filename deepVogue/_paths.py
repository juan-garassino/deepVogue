"""Runtime path resolution for deepVogue.

deepVogue is Colab-first: datasets and outputs live on Google Drive, never on
the local machine. Every path is read from an environment variable so the same
code runs unchanged in a notebook (Drive paths) and in local dry-runs (tmp
paths). Nothing here hardcodes a dataset location.

Multi-dataset layout: set ``DV_DATASET_NAME`` (e.g. ``tarot``, ``film``,
``tarot_pretrained``) and ``dataset_dir`` / ``run_dir`` / ``drive_sync`` /
``anchors_dir`` / ``walks_dir`` are all suffixed with that name so several
datasets can coexist under one Drive root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Paths:
    data_dir: Path        # raw stills or movie files (preprocessing input)
    dataset_dir: Path     # prepared StyleGAN zip lands here
    run_dir: Path         # training outputs (snapshots, .pkl, fakes)
    drive_sync: Optional[Path]  # mirror of run_dir (None disables sync)
    network_pkl: Optional[Path] # default checkpoint for inference scripts
    anchors_dir: Path     # projected W+ vectors per frame (latent cinema)
    walks_dir: Path       # rendered mp4s
    dataset_name: Optional[str] = None  # multi-dataset suffix


_ENV = {
    "data_dir":     "DV_DATA_DIR",
    "dataset_dir":  "DV_DATASET_DIR",
    "run_dir":      "DV_RUN_DIR",
    "drive_sync":   "DV_DRIVE_SYNC",
    "network_pkl":  "DV_NETWORK_PKL",
    "anchors_dir":  "DV_ANCHORS_DIR",
    "walks_dir":    "DV_WALKS_DIR",
    "dataset_name": "DV_DATASET_NAME",
}

# Path keys that get suffixed with ``<dataset_name>`` when it is set.
_DATASET_SCOPED = ("dataset_dir", "run_dir", "drive_sync", "anchors_dir", "walks_dir")


def _opt(env: str) -> Optional[Path]:
    v = os.environ.get(env)
    return Path(v) if v else None


def resolve(*, require: tuple[str, ...] = ()) -> Paths:
    """Read DV_* env vars. Caller passes which keys are required for its task."""
    raw = {k: _opt(v) for k, v in _ENV.items() if k != "dataset_name"}
    ds_name = os.environ.get("DV_DATASET_NAME") or None
    for key in require:
        if raw.get(key) is None:
            raise RuntimeError(
                f"deepVogue: ${_ENV[key]} is not set. "
                f"In Colab, set it before running this command "
                f"(e.g. os.environ['{_ENV[key]}'] = '/content/drive/MyDrive/deepVogue/...')."
            )

    def _scoped(base: Optional[Path], default: Path) -> Path:
        root = base or default
        return root / ds_name if ds_name else root

    return Paths(
        data_dir=raw["data_dir"] or Path("/tmp/deepVogue/data"),
        dataset_dir=_scoped(raw["dataset_dir"], Path("/tmp/deepVogue/dataset")),
        run_dir=_scoped(raw["run_dir"], Path("/tmp/deepVogue/run")),
        drive_sync=(raw["drive_sync"] / ds_name) if (raw["drive_sync"] and ds_name) else raw["drive_sync"],
        network_pkl=raw["network_pkl"],
        anchors_dir=_scoped(raw["anchors_dir"], Path("/tmp/deepVogue/anchors")),
        walks_dir=_scoped(raw["walks_dir"], Path("/tmp/deepVogue/walks")),
        dataset_name=ds_name,
    )


def env_summary() -> str:
    p = resolve()
    keys = ("data_dir", "dataset_dir", "run_dir", "drive_sync", "network_pkl",
            "anchors_dir", "walks_dir", "dataset_name")
    lines = [f"  {_ENV[k]:<18} = {getattr(p, k)}" for k in keys]
    return "deepVogue paths:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# checkpoint discovery (used by `make resume`, `make latest-pkl`, FastAPI registry)
# ---------------------------------------------------------------------------

_SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def _scan_pkls(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("network-snapshot-*.pkl"))


def list_runs(dataset_name: Optional[str] = None) -> list[Path]:
    """List run directories for a dataset. Looks in drive_sync first (durable),
    falls back to run_dir. Each entry is one run subdirectory (e.g.
    ``00000-stylegan3-t-tarot``).
    """
    name = dataset_name or os.environ.get("DV_DATASET_NAME") or None
    saved = os.environ.get("DV_DATASET_NAME")
    try:
        if name is not None:
            os.environ["DV_DATASET_NAME"] = name
        p = resolve()
        for root in (p.drive_sync, p.run_dir):
            if root and root.exists():
                runs = [d for d in root.iterdir() if d.is_dir()]
                if runs:
                    return sorted(runs)
        return []
    finally:
        if saved is None:
            os.environ.pop("DV_DATASET_NAME", None)
        else:
            os.environ["DV_DATASET_NAME"] = saved


def latest_snapshot(dataset_name: Optional[str] = None) -> Optional[Path]:
    """Newest ``network-snapshot-*.pkl`` across all runs for the dataset.

    Ranked by the kimg integer in the filename (NVIDIA's training_loop writes
    these monotonically), with mtime as tiebreaker. Returns None if no
    snapshots exist.
    """
    candidates: list[Path] = []
    for run in list_runs(dataset_name):
        candidates.extend(_scan_pkls(run))
    if not candidates:
        return None

    def _key(p: Path) -> tuple[int, float]:
        m = _SNAPSHOT_RE.search(p.name)
        kimg = int(m.group(1)) if m else 0
        return (kimg, p.stat().st_mtime)

    return max(candidates, key=_key)
