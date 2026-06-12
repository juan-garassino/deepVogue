"""Shared inference runtime — device pick, pkl→G_ema, frame/video output.

torch and imageio are imported lazily inside each function so `--help` on the
CLIs stays fast and non-GPU code paths (registry, paths) never pull them in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def pick_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(pkl: Union[str, Path], device=None):
    """Load G_ema from a StyleGAN ``.pkl`` onto ``device``: eval mode, grads off."""
    from deepVogue import legacy, neuronal_network_utils

    device = device or pick_device()
    with neuronal_network_utils.util.open_url(str(pkl)) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"]
    return G.requires_grad_(False).eval().to(device)


def to_uint8_hwc(img):
    """``G.synthesis`` output ``(1, C, H, W)`` in [-1, 1] → uint8 HWC numpy."""
    import torch

    img = (img + 1) * (255 / 2)
    return img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()


def open_video_writer(path: Union[str, Path], fps: int):
    import imageio.v2 as imageio

    # format="FFMPEG" so a missing imageio-ffmpeg backend raises immediately
    # instead of imageio silently falling back to a non-video plugin.
    return imageio.get_writer(
        str(path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        bitrate="16M",
        macro_block_size=1,
    )
