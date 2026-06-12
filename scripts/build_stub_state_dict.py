"""Build a tiny SG3-t generator state dict for local-nano testing.

Produces a structurally-valid SG3-t Generator that can forward-pass on CPU.
Run once; commit the output blob.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepVogue.training.networks_stylegan3 import Generator


def build(out_path: Path, img_resolution: int = 64, channel_max: int = 16) -> None:
    G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=img_resolution,
        img_channels=3,
        channel_base=4096,
        channel_max=channel_max,
        num_layers=4,
        first_cutoff=2,
        first_stopband=2.5,
    )
    G.eval()
    with torch.no_grad():
        z = torch.randn(1, 512)
        img = G(z, None)
        assert img.shape == (1, 3, img_resolution, img_resolution), img.shape
    torch.save({"G_ema": G.state_dict(), "stub": True}, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/stub_sg3_state_dict.pt"),
    )
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--cmax", type=int, default=16)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.res, args.cmax)
