"""StyleGAN ``.pkl`` LRU cache + generate/walk/factor forward wrappers."""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from typing import List, Optional

import fsspec
import numpy as np

from .registry import ModelEntry


class _Cache:
    """Tiny thread-safe LRU. Values are loaded G_ema modules on a device."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._items: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: str, loader):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = loader()  # outside lock; loading is slow
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return value


_CACHE = _Cache(capacity=2)


# ---------------------------------------------------------------------------
# .pkl → G_ema
# ---------------------------------------------------------------------------


def _load_pkl(pkl_uri: str):
    """Load a StyleGAN .pkl from any fsspec-supported URI (gs://, s3://, local);
    return G_ema on device, eval mode."""
    from deepVogue import legacy
    from deepVogue._runtime import pick_device

    device = pick_device()
    with fsspec.open(pkl_uri, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    return G.requires_grad_(False).eval().to(device)


def load(entry: ModelEntry):
    uri = entry.pkl_resolved or entry.pkl
    return _CACHE.get_or_load(entry.id, lambda: _load_pkl(uri))


# ---------------------------------------------------------------------------
# forward wrappers
# ---------------------------------------------------------------------------


def _seed_to_w(G, seed: int, trunc: float, device):
    import torch

    z = (
        torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
        .to(device)
        .float()
    )
    w = G.mapping(z, None, truncation_psi=trunc)
    return w


def generate(
    entry: ModelEntry,
    *,
    seed: int,
    trunc: Optional[float] = None,
    factor_idx: Optional[int] = None,
    factor_amp: float = 0.0,
) -> bytes:
    """Return PNG bytes for a single generated image."""
    import torch
    from PIL import Image
    from deepVogue._runtime import to_uint8_hwc

    G = load(entry)
    device = next(G.parameters()).device
    psi = entry.default_trunc if trunc is None else trunc
    with torch.no_grad():
        w = _seed_to_w(G, seed, psi, device)
        if factor_idx is not None and factor_amp != 0.0 and entry.factors:
            f = torch.load(entry.factors, map_location="cpu")
            eigvec = f["eigvec"]
            direction = eigvec[:, factor_idx].to(device).float()
            w = w + factor_amp * direction[None, None, :]
        img = G.synthesis(w, noise_mode="const")
    buf = io.BytesIO()
    Image.fromarray(to_uint8_hwc(img)).save(buf, format="PNG")
    return buf.getvalue()


def walk(
    entry: ModelEntry,
    *,
    seeds: List[int],
    steps: int,
    fps: int,
    mode: str = "cubic",
    trunc: Optional[float] = None,
) -> bytes:
    """Return mp4 bytes for a latent walk between seeds."""
    import os
    import tempfile
    import torch
    from deepVogue._runtime import open_video_writer, to_uint8_hwc
    from deepVogue.walk import _interpolate_anchors

    if len(seeds) < 2:
        raise ValueError("walk needs ≥2 seeds")
    G = load(entry)
    device = next(G.parameters()).device
    psi = entry.default_trunc if trunc is None else trunc
    with torch.no_grad():
        ws = [
            _seed_to_w(G, s, psi, device).cpu().numpy()[0]  # (num_ws, w_dim)
            for s in seeds
        ]
    anchors = np.stack(ws, axis=0)
    traj = _interpolate_anchors(anchors, frames_per_segment=steps, mode=mode)

    # libx264 muxing needs a seekable file, so go through a temp path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        writer = open_video_writer(tmp_path, fps)
        try:
            with torch.no_grad():
                for t in range(traj.shape[0]):
                    w = torch.from_numpy(traj[t : t + 1]).to(device).float()
                    img = G.synthesis(w, noise_mode="const")
                    writer.append_data(to_uint8_hwc(img))
        finally:
            writer.close()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
