"""DiT backbone with PixArt-alpha's adaLN-single conditioning.

Ported technique (diffusion-dit.md #1, "PixArt-alpha"):

- **adaLN-single** (PixArt.py:44-54, 203-210; PixArt_blocks.py:25-27): a *single*
  global ``t_block`` (SiLU + Linear) produces the 6 modulation vectors
  (shift/scale/gate for the attention branch and the MLP branch) shared across
  all blocks, plus a tiny per-block learnable ``scale_shift_table`` that
  perturbs them. This replaces DiT's per-block adaLN MLP and cuts ~27% of the
  parameters — the load-bearing simplification for a budget latent-cinema run.
- **zero-init output projections** so conditioning ramps in smoothly when
  fine-tuning off a class-conditional checkpoint (the 3-stage playbook).

The 3-stage *training* decomposition (pixel-dependency -> text/garment
alignment -> aesthetic finetune) is expressed as ``STAGES`` + the
``freeze_for_stage`` hook here, and orchestrated by ``training_loop_dit``.

Shape conventions (annotated inline, hkproj-style):
    x : (B, C, H, W)   VAE latents (C = latent channels, e.g. 4)
    t : (B,)           integer diffusion timesteps
    y : (B, D_cond)    optional pooled conditioning vector (class / text / CLIP)
    out : (B, C, H, W) predicted noise (epsilon-parameterization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 3-stage training decomposition (PixArt README.md:119). Orchestrated by
# training_loop_dit.DiTTrainer; referenced here so freeze_for_stage can gate
# which parameters train in each stage.
STAGES: tuple[str, ...] = ("pixel", "alignment", "aesthetic")

DIT_SCHEMA_VERSION = "1.0.0"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM-style modulation: x * (1 + scale) + shift, broadcast over tokens.

    x : (B, N, D)   shift/scale : (B, D) -> unsqueezed to (B, 1, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def adaln_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Public alias of :func:`modulate` (exported for tests / reuse)."""
    return modulate(x, shift, scale)


def _get_2d_sincos_pos_embed(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos positional embedding, (grid*grid, dim).

    Mirrors PixArt's fixed pos-embed (resolution transfer via lewei_scale is
    omitted at scaffold stage — a fixed grid is enough for shape tests).
    """
    assert dim % 4 == 0, "pos-embed dim must be divisible by 4"
    coords = np.arange(grid, dtype=np.float32)
    gy, gx = np.meshgrid(coords, coords, indexing="ij")

    def _embed_1d(pos: np.ndarray) -> np.ndarray:
        omega = np.arange(dim // 4, dtype=np.float32) / (dim / 4.0)
        omega = 1.0 / (10000**omega)  # (dim/4,)
        out = pos.reshape(-1)[:, None] * omega[None, :]  # (grid*grid, dim/4)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([_embed_1d(gy), _embed_1d(gx)], axis=1)  # (grid*grid, dim)
    return torch.from_numpy(emb).float()


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP embedding, (B,) -> (B, D)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _timestep_freqs(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -np.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, freq_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._timestep_freqs(t))  # (B, D)


class DiTBlock(nn.Module):
    """A DiT transformer block with adaLN-single modulation.

    The 6 modulation vectors (msa shift/scale/gate + mlp shift/scale/gate) are
    supplied *globally* by the model's shared t_block; each block only owns a
    small learnable ``scale_shift_table`` (1, 6, D) that perturbs them
    (PixArt_blocks.py:25-27).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # Per-block perturbation of the shared modulation (adaLN-single).
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, hidden_size))

    def forward(self, x: torch.Tensor, t6: torch.Tensor) -> torch.Tensor:
        # x : (B, N, D) ; t6 : (B, 6, D) global modulation from t_block
        mod = self.scale_shift_table + t6  # (B, 6, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.unbind(1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)  # (B, N, D)
        attn_out, _ = self.attn(h, h, h, need_weights=False)  # (B, N, D)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """adaLN-single final layer -> unpatchify projection (zero-init)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, hidden_size))
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        # zero-init: the model starts as the identity noise predictor.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.scale_shift_table + t2).unbind(1)  # each (B, D)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class DiTConfig:
    """Config for the DiT backbone. Defaults are the tiny "DiT-S/2"-ish shape
    that fits CPU shape-tests; the L4-scale config is set by training_loop_dit.
    """

    input_size: int = 32  # DV_DIT_INPUT_SIZE — latent H=W (e.g. 32 for 256px/8x VAE)
    patch_size: int = 2  # DV_DIT_PATCH — patchify stride
    in_channels: int = 4  # DV_DIT_IN_CH — VAE latent channels (SD-VAE = 4)
    hidden_size: int = 384  # DV_DIT_HIDDEN — S=384, B=768, XL=1152
    depth: int = 4  # DV_DIT_DEPTH — number of DiT blocks (PixArt-XL = 28)
    num_heads: int = 6  # DV_DIT_HEADS — attention heads
    mlp_ratio: float = 4.0  # feed-forward expansion
    cond_dim: Optional[int] = None  # pooled conditioning dim (None = unconditional)
    learn_sigma: bool = False  # if True, also predict variance (out_ch *= 2)


class DiT(nn.Module):
    """PixArt-style DiT with adaLN-single over VAE latents.

    Predicts epsilon on ``(B, C, H, W)`` latents. Conditioning is optional and,
    when present, is added to the timestep embedding before the shared
    ``t_block`` — mirroring PixArt's shared-modulation design.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.out_channels = config.in_channels * (2 if config.learn_sigma else 1)
        self.grid = config.input_size // config.patch_size
        self.num_patches = self.grid * self.grid
        D = config.hidden_size

        self.x_embed = nn.Conv2d(
            config.in_channels, D, config.patch_size, stride=config.patch_size
        )
        self.register_buffer(
            "pos_embed",
            _get_2d_sincos_pos_embed(D, self.grid).unsqueeze(0),  # (1, N, D)
            persistent=False,
        )
        self.t_embed = TimestepEmbedder(D)
        if config.cond_dim is not None:
            self.cond_proj = nn.Linear(config.cond_dim, D)
            # zero-init cross-conditioning so text/CLIP ramps in smoothly.
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.cond_proj = None

        # Shared adaLN-single modulation generator: (B, D) -> (B, 6*D).
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))

        self.blocks = nn.ModuleList(
            [DiTBlock(D, config.num_heads, config.mlp_ratio) for _ in range(config.depth)]
        )
        self.final = FinalLayer(D, config.patch_size, self.out_channels)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, N, patch*patch*out_ch) -> (B, out_ch, H, W)
        B = x.shape[0]
        p, g, c = self.config.patch_size, self.grid, self.out_channels
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, g * p, g * p)

    def freeze_for_stage(self, stage: str) -> None:
        """Gate which params train per 3-stage decomposition (a training hook).

        - ``pixel``     : everything trains (learn pixel dependency).
        - ``alignment`` : freeze the patch/pos backbone, train conditioning
                          + blocks (learn text/garment alignment).
        - ``aesthetic`` : train only the final layer + last block (short polish).
        """
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        for p in self.parameters():
            p.requires_grad_(True)
        if stage == "alignment":
            for p in self.x_embed.parameters():
                p.requires_grad_(False)
        elif stage == "aesthetic":
            for p in self.parameters():
                p.requires_grad_(False)
            for m in (self.final, self.blocks[-1]):
                for p in m.parameters():
                    p.requires_grad_(True)
        logger.debug("froze DiT params for stage=%s", stage)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x : (B, C, H, W) -> patch embed -> (B, N, D)
        h = self.x_embed(x).flatten(2).transpose(1, 2)
        h = h + self.pos_embed

        c = self.t_embed(t)  # (B, D)
        if self.cond_proj is not None and y is not None:
            c = c + self.cond_proj(y)

        t6 = self.t_block(c).reshape(-1, 6, self.config.hidden_size)  # (B, 6, D)
        t2 = t6[:, :2]  # final layer reuses the first 2 modulation vectors

        for block in self.blocks:
            h = block(h, t6)  # (B, N, D)

        h = self.final(h, t2)  # (B, N, patch*patch*out_ch)
        return self._unpatchify(h)  # (B, out_ch, H, W)
