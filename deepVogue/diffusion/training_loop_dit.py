"""3-stage DiT training orchestrator (SCAFFOLD — wired, not run).

Ported technique (diffusion-dit.md #1, "3-stage training decomposition",
PixArt README.md:119): decompose training into cheap-to-expensive stages ---

    1. ``pixel``     — learn pixel/motion dependency unconditionally (cheap).
    2. ``alignment`` — align to text/garment conditioning (freeze backbone).
    3. ``aesthetic`` — short aesthetic finetune (train only the head).

This is how PixArt hits near-SD quality at ~2% of the cost — the budget
playbook for a €450/mo latent-cinema run on garussino-ml.

This orchestrator is deliberately CPU-importable and shape-testable: the
optimizer step runs on a tiny random batch so ``run_stage`` / ``run_all`` are
exercisable without a GPU or a dataset, but REAL training (real VAE latents,
EMA, DDP, checkpointing to GCS) is a GPU-only concern and is NOT wired here.
See the RunPod validation checklist on the PR and NEEDS-GPU-VALIDATION in
CLAUDE.md.

Both backbones are supported:
    - image DiT  : batches are (B, C, H, W)     latents
    - video Latte: batches are (B, T, C, H, W)  latents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

from deepVogue.diffusion.dit import STAGES, DiT
from deepVogue.diffusion.latte import Latte

logger = logging.getLogger(__name__)

Backbone = Union[DiT, Latte]
BatchSampler = Callable[[], torch.Tensor]

# Per-stage kimg budgets (thousands of images). Overridable via DV_DIT_KIMG_*.
DEFAULT_STAGE_KIMG: dict[str, int] = {
    "pixel": 5000,
    "alignment": 2000,
    "aesthetic": 500,
}


@dataclass
class DiTTrainConfig:
    """Config for the 3-stage trainer (scaffold defaults are tiny)."""

    num_timesteps: int = 1000  # DV_DIT_TIMESTEPS — diffusion steps
    lr: float = 1e-4  # DV_DIT_LR
    stage_steps: dict[str, int] = field(
        default_factory=lambda: {"pixel": 2, "alignment": 2, "aesthetic": 2}
    )  # optimizer steps per stage — tiny for scaffold; real run uses DEFAULT_STAGE_KIMG
    ema_decay: float = 0.9999  # documented; EMA itself is a GPU-validation TODO
    device: str = "cpu"


def diffusion_loss(
    model: Backbone,
    x0: torch.Tensor,
    num_timesteps: int,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simple epsilon-prediction DDPM loss (uniform-t, unit-noise scaling).

    Deliberately schedule-light: a real run plugs in a cosine/linear beta
    schedule and v-prediction. This scaffold only needs a differentiable
    scalar so the optimizer step is exercisable in the shape-tests.

    x0 : (B, C, H, W) or (B, T, C, H, W)
    """
    B = x0.shape[0]
    t = torch.randint(0, num_timesteps, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    # scaffold forward process: linearly interpolate x0 -> noise by t fraction.
    frac = (t.float() / num_timesteps).view((B,) + (1,) * (x0.dim() - 1))
    x_t = (1 - frac) * x0 + frac * noise

    if isinstance(model, Latte):
        pred = model(x_t, t)  # (B, T, C, H, W)
    else:
        pred = model(x_t, t, cond)  # (B, C, H, W)
    return F.mse_loss(pred, noise)


class DiTTrainer:
    """Orchestrates the 3-stage decomposition over a single backbone."""

    def __init__(
        self,
        model: Backbone,
        config: Optional[DiTTrainConfig] = None,
    ):
        self.model = model
        self.config = config or DiTTrainConfig()
        self.model.to(self.config.device)
        self._is_video = isinstance(model, Latte)

    def _optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.lr)

    def run_stage(self, stage: str, sample_batch: BatchSampler) -> float:
        """Run one stage's optimizer steps; returns the last loss value.

        ``sample_batch`` yields a latent batch of the right rank for the
        backbone. ``freeze_for_stage`` gates which params train (DiT only;
        Latte trains fully in this scaffold).
        """
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected {STAGES}")
        if isinstance(self.model, DiT):
            self.model.freeze_for_stage(stage)
        opt = self._optimizer()
        self.model.train()

        steps = self.config.stage_steps.get(stage, 0)
        last = float("nan")
        for step in range(steps):
            x0 = sample_batch().to(self.config.device)
            loss = diffusion_loss(self.model, x0, self.config.num_timesteps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
            logger.info("stage=%s step=%d loss=%.4f", stage, step, last)
        return last

    def run_all(self, sample_batch: BatchSampler) -> dict[str, float]:
        """Run pixel -> alignment -> aesthetic in order; returns per-stage loss."""
        return {stage: self.run_stage(stage, sample_batch) for stage in STAGES}


def main() -> None:  # pragma: no cover - CLI entry for `make train-diffusion`
    """CLI entry. Intentionally refuses to run real training on CPU.

    Wired for `make train-diffusion`, which sets DV_DIT_ALLOW_CPU=0 by default.
    A real run happens on GPU (RunPod / Cloud Run L4); this guard prevents an
    accidental multi-hour CPU job. See the PR's RunPod validation checklist.
    """
    import argparse
    import os

    from deepVogue.diffusion.dit import DiTConfig
    from deepVogue.diffusion.latte import LatteConfig

    parser = argparse.ArgumentParser(description="deepVogue 3-stage DiT trainer")
    parser.add_argument(
        "--backbone", choices=("dit", "latte"), default="latte",
        help="image DiT or video Latte",
    )
    parser.add_argument("--device", default=os.environ.get("DV_DIT_DEVICE", "cpu"))
    args = parser.parse_args()

    allow_cpu = os.environ.get("DV_DIT_ALLOW_CPU", "0") == "1"
    if args.device == "cpu" and not allow_cpu:
        raise SystemExit(
            "Refusing to run diffusion training on CPU. This subsystem is "
            "GPU-only and NEEDS-GPU-VALIDATION. Set DV_DIT_ALLOW_CPU=1 to "
            "force a (slow, scaffold-only) CPU smoke run, or run on an L4/A10."
        )

    if args.backbone == "latte":
        model: Backbone = Latte(LatteConfig())
    else:
        model = DiT(DiTConfig())
    trainer = DiTTrainer(model, DiTTrainConfig(device=args.device))
    logger.warning(
        "Running SCAFFOLD training (random latents, no real dataset/EMA/GCS). "
        "This is not a real training run."
    )

    cfg = model.config
    if isinstance(model, Latte):
        def sampler() -> torch.Tensor:
            return torch.randn(2, cfg.num_frames, cfg.in_channels, cfg.input_size, cfg.input_size)
    else:
        def sampler() -> torch.Tensor:
            return torch.randn(2, cfg.in_channels, cfg.input_size, cfg.input_size)

    results = trainer.run_all(sampler)
    logger.info("scaffold stage losses: %s", results)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
