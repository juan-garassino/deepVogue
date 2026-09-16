"""VAE latent encode/decode + cached-feature handling.

Ported technique (diffusion-dit.md #1 companion, "pre-extracted features"):
the training loop should only ever touch pre-extracted VAE latents cached to
disk (PixArt tools/extract_features.py + train.py:61-78). This module provides:

- :class:`LatentVAE` — a small, self-contained convolutional VAE that
  encodes ``(B, 3, H, W)`` images to ``(B, C, H/f, W/f)`` latents and back.
  It is CPU-importable and shape-correct; it is NOT the SD-VAE (a real run
  swaps in ``diffusers.AutoencoderKL`` — see ``from_pretrained`` stub). The
  point of the scaffold is that the *interface* (encode/decode/roundtrip
  shapes, scaling factor) matches what the DiT/Latte trainers expect.
- :func:`encode_and_cache` / :func:`load_cached` — write/read latent tensors
  to a cache dir so ``training_loop_dit`` reads latents, never pixels.

Shape conventions:
    image  : (B, 3, H, W)                     pixels in [-1, 1]
    latent : (B, C, H/factor, W/factor)       scaled by config.scaling_factor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

VAE_SCHEMA_VERSION = "1.0.0"


@dataclass
class VAEConfig:
    """Config for the latent VAE. Defaults mimic SD-VAE's 8x / 4-channel shape."""

    in_channels: int = 3  # RGB input
    latent_channels: int = 4  # DV_VAE_LATENT_CH — SD-VAE = 4
    downscale_factor: int = 8  # DV_VAE_FACTOR — spatial compression (must be power of 2)
    base_channels: int = 32  # width of the conv stack (tiny for scaffold)
    scaling_factor: float = 0.18215  # SD-VAE latent scaling constant


class _DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentVAE(nn.Module):
    """A minimal conv VAE with the SD-VAE latent interface.

    Encodes to a diagonal-Gaussian posterior (mean, logvar); ``encode`` returns
    the *scaled* sampled latent by default (what the diffusion trainer wants).
    """

    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        self.config = config or VAEConfig()
        f = self.config.downscale_factor
        if f & (f - 1) != 0:
            raise ValueError("downscale_factor must be a power of 2")
        n_stages = f.bit_length() - 1  # log2(f)
        base = self.config.base_channels

        # encoder: n_stages of stride-2 downsampling, then to 2*latent (mu/logvar)
        enc_layers: list[nn.Module] = [
            nn.Conv2d(self.config.in_channels, base, 3, padding=1)
        ]
        c = base
        for _ in range(n_stages):
            enc_layers.append(_DownBlock(c, c * 2))
            c *= 2
        enc_layers.append(nn.Conv2d(c, 2 * self.config.latent_channels, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder: latent -> n_stages of upsampling -> RGB
        dec_layers: list[nn.Module] = [
            nn.Conv2d(self.config.latent_channels, c, 3, padding=1)
        ]
        for _ in range(n_stages):
            dec_layers.append(_UpBlock(c, c // 2))
            c //= 2
        dec_layers.append(nn.Conv2d(c, self.config.in_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """(B, 3, H, W) -> scaled latent (B, C, H/f, W/f)."""
        h = self.encoder(image)
        mu, logvar = h.chunk(2, dim=1)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z * self.config.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Scaled latent (B, C, h, w) -> image (B, 3, H, W)."""
        z = latent / self.config.scaling_factor
        return self.decoder(z)

    def roundtrip(self, image: torch.Tensor) -> torch.Tensor:
        """encode(sample=False) -> decode; used for the shape/sanity check."""
        return self.decode(self.encode(image, sample=False))

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "LatentVAE":  # pragma: no cover
        """GPU-only: swap in a real ``diffusers.AutoencoderKL``.

        Intentionally not wired for CPU shape-tests. On a RunPod/L4 box this
        loads the real SD-VAE so cached latents match a production checkpoint.
        See NEEDS-GPU-VALIDATION in CLAUDE.md.
        """
        raise NotImplementedError(
            "from_pretrained requires diffusers + GPU; not part of the CPU scaffold"
        )


def encode_and_cache(
    vae: LatentVAE,
    images: torch.Tensor,
    cache_path: str | Path,
    *,
    sample: bool = False,
) -> Path:
    """Encode a batch and persist latents to ``cache_path`` (a .pt file).

    The trainer then reads only these tensors — never pixels — mirroring
    PixArt's pre-extracted-feature loop.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        latents = vae.encode(images, sample=sample)
    torch.save(
        {"latents": latents.cpu(), "schema": VAE_SCHEMA_VERSION}, cache_path
    )
    logger.info("cached %d latents -> %s", latents.shape[0], cache_path)
    return cache_path


def load_cached(cache_path: str | Path) -> torch.Tensor:
    """Load a cached latent tensor written by :func:`encode_and_cache`."""
    payload = torch.load(Path(cache_path), map_location="cpu")
    return payload["latents"]
