"""deepVogue latent-diffusion subsystem (parallel to the StyleGAN3 engine).

This is a NEW, additive subsystem that lives alongside — not in place of — the
vendored StyleGAN3 stack in ``deepVogue/training/``. It ports the ranked top-3
techniques from the diffusion/DiT reference analysis for the latent-cinema
direction:

- ``dit``   — PixArt-alpha DiT backbone with adaLN-single + the 3-stage training
              decomposition hooks (the *training strategy*).
- ``latte`` — Latte variant-1 spatio-temporal attention factorization with the
              per-axis timestep broadcast (the *video backbone*).
- ``vae_latent`` — VAE latent encode/decode + cached-feature handling (so the
              training loop only ever touches pre-extracted latents).
- ``training_loop_dit`` — the 3-stage train orchestrator (scaffold; wired,
              not run — see NEEDS-GPU-VALIDATION in CLAUDE.md).

Everything here is CPU-importable and shape-testable. Real training is
GPU-only and unvalidated; see the RunPod validation checklist on the PR.
"""

from __future__ import annotations

from deepVogue.diffusion.dit import DiT, DiTConfig, adaln_modulate
from deepVogue.diffusion.latte import Latte, LatteConfig
from deepVogue.diffusion.vae_latent import LatentVAE, VAEConfig

__all__ = [
    "DiT",
    "DiTConfig",
    "adaln_modulate",
    "Latte",
    "LatteConfig",
    "LatentVAE",
    "VAEConfig",
]
