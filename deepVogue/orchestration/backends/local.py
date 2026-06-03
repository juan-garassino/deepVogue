"""Local-nano backend."""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch

from deepVogue.clients import get_artifact_fs

log = logging.getLogger(__name__)


# ----- real -----


def prepare(
    *,
    source_uri: str,
    dataset_name: str,
    res: int,
    kind: str = "stills",
    target_uri: str,
    fps: int | None = None,
) -> dict[str, Any]:
    """Read source_uri (a local dir of images or video file path), produce dataset.zip on target_uri."""
    fs = get_artifact_fs()
    src = Path(source_uri)
    images: list[tuple[str, bytes]] = []

    if kind == "stills":
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = Image.open(p).convert("RGB").resize((res, res))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append((p.stem + ".png", buf.getvalue()))
    elif kind == "frames":
        raise NotImplementedError("frames prep needs ffmpeg; use `make prepare-frames`")
    else:
        raise ValueError(f"unknown kind: {kind}")

    if not images:
        raise RuntimeError(f"no images found under {src}")

    target_uri = target_uri.rstrip("/")
    proto, path = (
        (target_uri.split("://", 1) + [""])[:2]
        if "://" in target_uri
        else (None, target_uri)
    )
    target_path = f"/{path}" if proto else target_uri
    zip_path = f"{target_path}/{dataset_name}.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in images:
            zf.writestr(name, data)
    with fs.open(zip_path, "wb") as f:
        f.write(buf.getvalue())

    dataset_uri = f"{target_uri}/{dataset_name}.zip" if proto else zip_path
    return {"dataset_uri": dataset_uri, "n_images": len(images)}


def publish(**kw) -> dict[str, Any]:
    """Real publish — defers to deepVogue.publish."""
    from deepVogue.publish import publish_checkpoint

    src_dir = kw.pop("src_dir")
    return publish_checkpoint(src_dir=Path(src_dir), **kw)


# ----- mock -----


def _stub_state_dict() -> dict[str, Any]:
    fp = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "stub_sg3_state_dict.pt"
    )
    if not fp.exists():
        raise FileNotFoundError(
            f"stub state dict missing at {fp} — run `python scripts/build_stub_state_dict.py`"
        )
    return torch.load(fp, map_location="cpu", weights_only=False)


def train(
    *,
    dataset_name: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int = 64,
    target_uri: str | None = None,
    **_,
) -> dict[str, Any]:
    log.info("[nano-mock] training %s on %s for %d kimg", cfg, dataset_name, kimg)
    time.sleep(min(kimg / 50.0, 5.0))
    fs = get_artifact_fs()
    target_uri = (
        target_uri or f"memory://deepvogue-models/{dataset_name}_nano"
    ).rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    pkl_path = f"{target_path}/network-snapshot-{kimg:06d}.pkl"
    state = _stub_state_dict()
    buf = io.BytesIO()
    torch.save(state, buf)
    with fs.open(pkl_path, "wb") as f:
        f.write(buf.getvalue())
    fid = max(5.0, 200.0 - kimg / 25.0)  # fake monotonic decrease
    pkl_uri = f"{target_uri}/network-snapshot-{kimg:06d}.pkl" if proto else pkl_path
    return {"pkl": pkl_uri, "fid": fid, "kimg": kimg}


def project(
    *,
    model_id: str,
    frames_uri: str,
    stride: int = 4,
    steps: int = 50,
    target_uri: str,
    **_,
) -> dict[str, Any]:
    log.info("[nano-mock] project %s stride=%d steps=%d", model_id, stride, steps)
    time.sleep(1.0)
    fs = get_artifact_fs()
    target_uri = target_uri.rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    out_uri = f"{target_path}/{model_id}/0/projected_w.npz"
    arr = np.zeros((1, 16, 512), dtype=np.float32)
    buf = io.BytesIO()
    np.savez(buf, w=arr)
    with fs.open(out_uri, "wb") as f:
        f.write(buf.getvalue())
    return {"anchors_uri": f"{target_uri}/{model_id}", "n_anchors": 1}


def walk(
    *,
    model_id: str,
    target_uri: str,
    steps: int = 60,
    fps: int = 24,
    seeds: list[int] | None = None,
    anchors_uri: str | None = None,
    mode: str = "cubic",
    **_,
) -> dict[str, Any]:
    log.info("[nano-mock] walk %s steps=%d fps=%d mode=%s", model_id, steps, fps, mode)
    time.sleep(1.0)
    fs = get_artifact_fs()
    target_uri = target_uri.rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    walk_id = f"walk_{int(time.time())}"
    out = f"{target_path}/{model_id}/{walk_id}.mp4"
    with fs.open(out, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypmp42")  # minimal MP4 header bytes
    return {
        "walk_uri": f"{target_uri}/{model_id}/{walk_id}.mp4",
        "walk_id": walk_id,
    }


def eval(*, model_id: str, dataset_uri: str, **_) -> dict[str, Any]:
    log.info("[nano-mock] eval %s on %s", model_id, dataset_uri)
    time.sleep(0.5)
    return {"fid50k_full": 42.0, "kid50k_full": 0.003}
