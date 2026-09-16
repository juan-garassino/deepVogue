"""Latte — factorized spatio-temporal DiT (variant 1) for latent *video*.

Ported technique (diffusion-dit.md #2, "Latte"):

- **Variant-1 factorization** (latte.py:345-368): alternate spatial and
  temporal transformer blocks. Spatial blocks attend within a frame across
  patches; temporal blocks attend across frames at a fixed patch position.
  The whole trick is the rearrange dance between the two axes (kept as plain
  ``torch`` reshape/permute here to avoid an einops dependency):
      spatial : tokens are (B*F, N, D)   — attend over N patches
      temporal: tokens are (B*N, F, D)   — attend over F frames
- **Per-axis timestep broadcast** (latte.py:333-343): the (B, D) timestep
  modulation is repeated to (B*F, D) for spatial blocks and (B*N, D) for
  temporal blocks. Getting this broadcast right is the subtle bit everyone
  gets wrong when factorizing — it is unit-tested here.
- **Joint image-video hook** (latte_img.py, ``use_image_num``): the forward
  accepts ``image_num`` standalone frames appended to the video; temporal
  attention is masked for them. Scaffolded as a flag; the mask wiring is a
  GPU-validation TODO (see CLAUDE.md).

Reuses :class:`deepVogue.diffusion.dit.DiTBlock` / ``t_block`` so both models
share the adaLN-single machinery.

Shape conventions:
    x : (B, T, C, H, W)   latent video (T frames of C-channel latents)
    t : (B,)              diffusion timesteps
    out : (B, T, C, H, W) predicted noise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from deepVogue.diffusion.dit import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    _get_2d_sincos_pos_embed,
)

logger = logging.getLogger(__name__)


@dataclass
class LatteConfig:
    """Config for the Latte spatio-temporal DiT. Defaults = tiny "Latte-S/2"."""

    input_size: int = 32  # DV_LATTE_INPUT_SIZE — latent H=W
    patch_size: int = 2  # DV_LATTE_PATCH
    in_channels: int = 4  # DV_LATTE_IN_CH — VAE latent channels
    num_frames: int = 8  # DV_LATTE_FRAMES — T (temporal length)
    hidden_size: int = 384  # DV_LATTE_HIDDEN — S=384
    depth: int = 4  # DV_LATTE_DEPTH — total blocks; must be even (spatial+temporal pairs)
    num_heads: int = 6  # DV_LATTE_HEADS
    mlp_ratio: float = 4.0
    learn_sigma: bool = False


class Latte(nn.Module):
    """Variant-1 spatio-temporal DiT.

    Blocks alternate spatial / temporal. ``depth`` counts *all* blocks and must
    be even; blocks[0::2] are spatial, blocks[1::2] temporal (interleaved pairs,
    matching Latte's ``self.blocks[i:i+2]`` pairing).
    """

    def __init__(self, config: LatteConfig):
        super().__init__()
        if config.depth % 2 != 0:
            raise ValueError("Latte depth must be even (spatial/temporal pairs)")
        self.config = config
        self.out_channels = config.in_channels * (2 if config.learn_sigma else 1)
        self.grid = config.input_size // config.patch_size
        self.num_patches = self.grid * self.grid
        D = config.hidden_size

        self.x_embed = nn.Conv2d(
            config.in_channels, D, config.patch_size, stride=config.patch_size
        )
        # Fixed 2D spatial pos-embed + learned 1D temporal embed (added once).
        self.register_buffer(
            "pos_embed",
            _get_2d_sincos_pos_embed(D, self.grid).unsqueeze(0),  # (1, N, D)
            persistent=False,
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, config.num_frames, D)  # (1, T, D)
        )
        self.t_embed = TimestepEmbedder(D)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))

        self.blocks = nn.ModuleList(
            [DiTBlock(D, config.num_heads, config.mlp_ratio) for _ in range(config.depth)]
        )
        self.final = FinalLayer(D, config.patch_size, self.out_channels)

    def _unpatchify(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        # x : (B*F, N, patch*patch*out_ch) -> (B, F, out_ch, H, W)
        p, g, c = self.config.patch_size, self.grid, self.out_channels
        x = x.reshape(-1, g, g, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, g * p, g * p)  # (B*F, C, H, W)
        return x.reshape(-1, frames, c, g * p, g * p)  # (B, F, C, H, W)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        image_num: int = 0,
    ) -> torch.Tensor:
        # x : (B, T, C, H, W)
        B, T, C, H, W = x.shape
        N = self.num_patches
        D = self.config.hidden_size

        # patch-embed every frame independently -> (B*T, N, D)
        x = x.reshape(B * T, C, H, W)  # (b f) c h w
        h = self.x_embed(x).flatten(2).transpose(1, 2)  # (B*T, N, D)
        h = h + self.pos_embed  # spatial pos-embed broadcast over frames

        # add temporal embed once, per patch position (latte.py adds before
        # the first temporal block).
        temp = self.temp_embed[:, :T]  # (1, T, D)
        h = h.reshape(B, T, N, D) + temp.unsqueeze(2)  # (B, T, N, D)
        h = h.reshape(B * T, N, D)  # (B*T, N, D)

        # Global adaLN-single modulation from timestep, (B, 6, D).
        t6_base = self.t_block(self.t_embed(t)).reshape(B, 6, D)  # (B, 6, D)

        def _broadcast(mod: torch.Tensor, factor: int) -> torch.Tensor:
            # (B, S, D) -> (B*factor, S, D) with frames/patches as the fast axis,
            # i.e. row order (b0f0, b0f1, ..., b0f{factor-1}, b1f0, ...). This is
            # the per-axis timestep broadcast (latte.py:333-343).
            S = mod.shape[1]
            return (
                mod.unsqueeze(1)
                .expand(B, factor, S, D)
                .reshape(B * factor, S, D)
            )

        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # SPATIAL: tokens (B*T, N, D); broadcast timestep over frames.
                t6 = _broadcast(t6_base, T)  # (B*T, 6, D)
                h = block(h, t6)  # attend over N patches
            else:
                # TEMPORAL: rearrange to (B*N, T, D); broadcast over patches.
                # (B*T, N, D) -> (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
                h = (
                    h.reshape(B, T, N, D)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, T, D)
                )
                t6 = _broadcast(t6_base, N)  # (B*N, 6, D)
                h = block(h, t6)  # attend over T frames
                # back: (B*N, T, D) -> (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
                h = (
                    h.reshape(B, N, T, D)
                    .permute(0, 2, 1, 3)
                    .reshape(B * T, N, D)
                )

        t2 = _broadcast(t6_base[:, :2], T)  # (B*T, 2, D)
        h = self.final(h, t2)  # (B*T, N, patch*patch*out_ch)
        return self._unpatchify(h, frames=T)  # (B, T, out_ch, H, W)
