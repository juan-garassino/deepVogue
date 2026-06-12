"""Latent-cinema runtime.

Reads anchor W+ vectors produced by projector.py / pbaylies_projector.py,
interpolates between them in W+ space, and renders an mp4. This is the artifact
that turns deepVogue into latent cinema: a film replays itself through the
StyleGAN's manifold.

Anchors layout:
  <anchors_dir>/<id>/projected_w.npz   # key 'w' shape (1, num_ws, w_dim)

Order is taken from --order frames_index.json (kept_index → file) when
provided, else by alphabetical anchor id.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import click
import numpy as np


# ---------------------------------------------------------------------------
# noise trajectories — reuse the OpenSimplex generator from generate.py
# ---------------------------------------------------------------------------

def _osn_noise(num_frames: int, dim: int, *, seed: int = 0, diameter: float = 2.0) -> np.ndarray:
    """Smooth OSN-driven noise trajectory of shape (num_frames, dim)."""
    from deepVogue.generate import OSN
    rng = np.random.RandomState(seed)
    osns = [OSN(seed=int(rng.randint(0, 2**31 - 1)), diameter=diameter) for _ in range(dim)]
    out = np.zeros((num_frames, dim), dtype=np.float32)
    for f in range(num_frames):
        angle = (f / num_frames) * 2 * math.pi
        for d, osn in enumerate(osns):
            out[f, d] = osn.get_val(angle)
    return out


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------

def _slerp(t: float, a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    a_n = a / (np.linalg.norm(a) + eps)
    b_n = b / (np.linalg.norm(b) + eps)
    dot = np.clip(np.sum(a_n * b_n), -1.0, 1.0)
    if dot > 0.9995:
        return (1 - t) * a + t * b
    omega = math.acos(dot)
    s = math.sin(omega)
    return (math.sin((1 - t) * omega) / s) * a + (math.sin(t * omega) / s) * b


def _catmull_rom(t: float, p0: np.ndarray, p1: np.ndarray,
                 p2: np.ndarray, p3: np.ndarray,
                 tension: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom segment p1→p2 with neighbour anchors p0, p3.

    Smooth C¹ across segments, passes through every anchor exactly, no
    overshoot beyond the data hull for typical configurations.
    """
    t2, t3 = t * t, t * t * t
    return (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    ) * tension


def _interpolate_anchors(anchors: np.ndarray, frames_per_segment: int,
                         mode: str) -> np.ndarray:
    """anchors: (N, num_ws, w_dim) → trajectory: (T, num_ws, w_dim)."""
    if anchors.shape[0] < 2:
        raise click.ClickException(f"need ≥2 anchors, got {anchors.shape[0]}")
    out: List[np.ndarray] = []
    n = anchors.shape[0]
    for i in range(n - 1):
        a, b = anchors[i], anchors[i + 1]
        for f in range(frames_per_segment):
            t = f / frames_per_segment
            if mode == "linear":
                out.append((1 - t) * a + t * b)
            elif mode == "slerp":
                # slerp per row (W+ has num_ws rows; each row lives on its own sphere)
                rows = [_slerp(t, a[r], b[r]) for r in range(a.shape[0])]
                out.append(np.stack(rows, axis=0))
            elif mode == "bezier":
                # quadratic bezier with the segment midpoint pulled toward avg of neighbours
                if i == 0 or i == n - 2:
                    ctrl = (a + b) * 0.5
                else:
                    prev, nxt = anchors[i - 1], anchors[i + 2]
                    ctrl = ((a + b) * 0.5) + 0.25 * ((b - prev) + (a - nxt)) * 0.5
                out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b)
            elif mode == "cubic":
                # Catmull-Rom; clamp endpoints by duplicating outer anchors
                p0 = anchors[i - 1] if i > 0 else a
                p3 = anchors[i + 2] if i + 2 < n else b
                out.append(_catmull_rom(t, p0, a, b, p3))
            else:
                raise click.ClickException(f"unknown interp mode: {mode}")
    out.append(anchors[-1])
    return np.stack(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# anchor loading
# ---------------------------------------------------------------------------

def _load_anchors(anchors_dir: Path, order_json: Optional[Path]) -> np.ndarray:
    files: List[Path] = []
    if order_json and order_json.exists():
        idx = json.loads(order_json.read_text())
        for frame in idx["frames"]:
            kid = f"{frame['kept_index']:08d}"
            cand = anchors_dir / kid / "projected_w.npz"
            if cand.exists():
                files.append(cand)
    if not files:
        files = sorted(anchors_dir.rglob("projected_w.npz"))
    if not files:
        raise click.ClickException(f"no projected_w.npz under {anchors_dir}")
    ws = []
    for f in files:
        w = np.load(f)["w"]
        if w.ndim == 3:
            w = w[0]
        ws.append(w)
    return np.stack(ws, axis=0)  # (N, num_ws, w_dim)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, type=click.Path(exists=True))
@click.option("--anchors", "anchors_dir", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--order", "order_json", type=click.Path(path_type=Path), default=None,
              help="frames_index.json from deepvogue-prepare frames")
@click.option("--out", "out_mp4", required=True, type=click.Path(path_type=Path))
@click.option("--interp", type=click.Choice(["linear", "slerp", "bezier", "cubic"]), default="cubic")
@click.option("--fps", type=int, default=24, show_default=True)
@click.option("--frames-per-segment", type=int, default=24, show_default=True,
              help="frames rendered between consecutive anchors")
@click.option("--noise-mode", type=click.Choice(["const", "osn", "reactive"]), default="const")
@click.option("--audio", type=click.Path(path_type=Path), default=None,
              help="audio file for --noise-mode reactive")
@click.option("--factor-drift", type=click.Path(path_type=Path), default=None,
              help="factors.pt from closed_form_factorization.py")
@click.option("--factor-index", type=int, default=0)
@click.option("--factor-amp", type=float, default=0.0)
@click.option("--seed", type=int, default=0)
def main(network_pkl: str, anchors_dir: Path, order_json: Optional[Path],
         out_mp4: Path, interp: str, fps: int, frames_per_segment: int,
         noise_mode: str, audio: Optional[Path],
         factor_drift: Optional[Path], factor_index: int, factor_amp: float,
         seed: int) -> None:
    """Render a latent walk through projected anchors as an mp4."""
    import torch
    from deepVogue import _runtime

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    device = _runtime.pick_device()

    click.echo(f"[walk] loading {network_pkl}")
    G = _runtime.load_generator(network_pkl, device)

    anchors = _load_anchors(anchors_dir, order_json)
    click.echo(f"[walk] {anchors.shape[0]} anchors, W+ shape {anchors.shape[1:]}")

    traj = _interpolate_anchors(anchors, frames_per_segment, interp)  # (T, num_ws, w_dim)
    T = traj.shape[0]
    click.echo(f"[walk] {T} output frames @ {fps} fps  ({T/fps:.1f}s)")

    # optional SeFa factor drift
    if factor_drift and factor_amp != 0.0:
        f = torch.load(str(factor_drift), map_location="cpu")
        eigvecs = f["eigvec"].numpy() if hasattr(f["eigvec"], "numpy") else np.asarray(f["eigvec"])
        direction = eigvecs[:, factor_index]
        ramp = np.linspace(0.0, factor_amp, T, dtype=np.float32)
        traj = traj + ramp[:, None, None] * direction[None, None, :]
        click.echo(f"[walk] applied factor {factor_index} with amp ramp 0→{factor_amp}")

    # noise modulation (optional shimmer / audio reactivity)
    noise_mod = None
    if noise_mode == "osn":
        noise_mod = _osn_noise(T, traj.shape[2], seed=seed)
    elif noise_mode == "reactive":
        if audio is None:
            raise click.ClickException("--noise-mode reactive requires --audio")
        try:
            import librosa
        except ImportError as e:
            raise click.ClickException("install librosa for audio-reactive walks") from e
        y, sr = librosa.load(str(audio), sr=None, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=max(1, len(y) // T))[0]
        rms = (rms - rms.min()) / (rms.ptp() + 1e-7)
        rms = np.interp(np.linspace(0, len(rms) - 1, T), np.arange(len(rms)), rms)
        base = _osn_noise(T, traj.shape[2], seed=seed)
        noise_mod = base * (0.4 + 1.6 * rms[:, None])

    writer = _runtime.open_video_writer(out_mp4, fps)
    try:
        with torch.no_grad():
            for t in range(T):
                w = torch.from_numpy(traj[t:t + 1]).to(device)  # (1, num_ws, w_dim)
                if noise_mod is not None:
                    w = w + torch.from_numpy(noise_mod[t][None, None, :]).to(device) * 0.05
                img = G.synthesis(w, noise_mode="const")
                writer.append_data(_runtime.to_uint8_hwc(img))
                if (t + 1) % max(1, T // 20) == 0:
                    click.echo(f"  frame {t+1}/{T}")
    finally:
        writer.close()
    click.echo(f"✓ {out_mp4}")


if __name__ == "__main__":
    main()
