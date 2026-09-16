"""Shape / import tests for the deepVogue latent-diffusion subsystem.

CPU-only, no real training. Guards the port of the ranked top-3 techniques
(diffusion-dit.md): PixArt adaLN-single DiT, Latte variant-1 spatio-temporal
factorization, and the VAE latent interface. These tests verify shapes and
that the wired 3-stage orchestrator runs an optimizer step — they do NOT
validate training quality (that is GPU-only; see NEEDS-GPU-VALIDATION).
"""

from __future__ import annotations

import torch

from deepVogue.diffusion import DiT, DiTConfig, Latte, LatteConfig, LatentVAE, VAEConfig
from deepVogue.diffusion.dit import STAGES, adaln_modulate
from deepVogue.diffusion.training_loop_dit import (
    DiTTrainConfig,
    DiTTrainer,
    diffusion_loss,
)


# --- tiny configs that fit CPU ---------------------------------------------

def _dit() -> DiT:
    return DiT(
        DiTConfig(
            input_size=16, patch_size=2, in_channels=4,
            hidden_size=48, depth=2, num_heads=4,
        )
    )


def _cond_dit() -> DiT:
    return DiT(
        DiTConfig(
            input_size=16, patch_size=2, in_channels=4,
            hidden_size=48, depth=2, num_heads=4, cond_dim=32,
        )
    )


def _latte() -> Latte:
    return Latte(
        LatteConfig(
            input_size=16, patch_size=2, in_channels=4, num_frames=3,
            hidden_size=48, depth=2, num_heads=4,
        )
    )


def _vae() -> LatentVAE:
    return LatentVAE(VAEConfig(latent_channels=4, downscale_factor=4, base_channels=8))


# --- imports ----------------------------------------------------------------

def test_public_api_imports():
    import deepVogue.diffusion as d

    for name in ("DiT", "DiTConfig", "Latte", "LatteConfig", "LatentVAE", "VAEConfig"):
        assert hasattr(d, name)


# --- DiT --------------------------------------------------------------------

def test_dit_forward_shape():
    m = _dit()
    x = torch.randn(2, 4, 16, 16)  # (B, C, H, W)
    t = torch.randint(0, 1000, (2,))
    out = m(x, t)
    assert out.shape == x.shape  # epsilon prediction, same shape


def test_dit_conditional_forward_shape():
    m = _cond_dit()
    x = torch.randn(3, 4, 16, 16)
    t = torch.randint(0, 1000, (3,))
    y = torch.randn(3, 32)  # pooled conditioning
    out = m(x, t, y)
    assert out.shape == x.shape


def test_dit_learn_sigma_doubles_channels():
    m = DiT(
        DiTConfig(
            input_size=16, patch_size=2, in_channels=4,
            hidden_size=48, depth=1, num_heads=4, learn_sigma=True,
        )
    )
    x = torch.randn(1, 4, 16, 16)
    out = m(x, torch.randint(0, 1000, (1,)))
    assert out.shape == (1, 8, 16, 16)  # 4 in-channels * 2


def test_adaln_modulate_broadcasts_over_tokens():
    x = torch.randn(2, 5, 8)  # (B, N, D)
    shift = torch.randn(2, 8)
    scale = torch.randn(2, 8)
    out = adaln_modulate(x, shift, scale)
    assert out.shape == x.shape


def test_dit_freeze_for_stage_gates_params():
    m = _dit()
    m.freeze_for_stage("aesthetic")  # only final + last block trainable
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("final") for n in trainable)
    # x_embed backbone must be frozen in the aesthetic stage
    assert all(not n.startswith("x_embed") for n in trainable)


# --- Latte ------------------------------------------------------------------

def test_latte_forward_shape():
    m = _latte()
    x = torch.randn(2, 3, 4, 16, 16)  # (B, T, C, H, W)
    t = torch.randint(0, 1000, (2,))
    out = m(x, t)
    assert out.shape == x.shape


def test_latte_variable_batch_and_frames():
    m = Latte(
        LatteConfig(
            input_size=16, patch_size=2, in_channels=4, num_frames=5,
            hidden_size=48, depth=4, num_heads=4,
        )
    )
    x = torch.randn(1, 5, 4, 16, 16)
    out = m(x, torch.randint(0, 1000, (1,)))
    assert out.shape == x.shape


def test_latte_odd_depth_rejected():
    import pytest

    with pytest.raises(ValueError):
        Latte(LatteConfig(depth=3))


# --- VAE --------------------------------------------------------------------

def test_vae_encode_downscales():
    vae = _vae()
    img = torch.randn(2, 3, 32, 32)
    z = vae.encode(img)
    assert z.shape == (2, 4, 8, 8)  # 32 / downscale_factor(4)


def test_vae_roundtrip_shape():
    vae = _vae()
    img = torch.randn(2, 3, 32, 32)
    out = vae.roundtrip(img)
    assert out.shape == img.shape


def test_vae_cache_roundtrip(tmp_path):
    from deepVogue.diffusion.vae_latent import encode_and_cache, load_cached

    vae = _vae()
    img = torch.randn(3, 3, 32, 32)
    path = encode_and_cache(vae, img, tmp_path / "feat.pt")
    latents = load_cached(path)
    assert latents.shape == (3, 4, 8, 8)


# --- 3-stage training orchestrator (scaffold, wired-not-run) ----------------

def test_diffusion_loss_is_scalar_and_backprops():
    m = _dit()
    x0 = torch.randn(2, 4, 16, 16)
    loss = diffusion_loss(m, x0, num_timesteps=1000)
    assert loss.dim() == 0
    loss.backward()  # gradients flow


def test_trainer_runs_all_three_stages():
    m = _dit()
    trainer = DiTTrainer(
        m, DiTTrainConfig(stage_steps={s: 1 for s in STAGES}, num_timesteps=100)
    )

    def sampler() -> torch.Tensor:
        return torch.randn(2, 4, 16, 16)

    results = trainer.run_all(sampler)
    assert set(results) == set(STAGES)
    assert all(isinstance(v, float) for v in results.values())


def test_trainer_runs_latte_stage():
    m = _latte()
    trainer = DiTTrainer(
        m, DiTTrainConfig(stage_steps={"pixel": 1}, num_timesteps=100)
    )

    def sampler() -> torch.Tensor:
        return torch.randn(2, 3, 4, 16, 16)  # (B, T, C, H, W)

    last = trainer.run_stage("pixel", sampler)
    assert isinstance(last, float)
