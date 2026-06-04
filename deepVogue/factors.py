"""SeFa factor discover/apply wrapper.

Thin Drive-aware CLI on top of deepVogue/closed_form_factorization.py and
deepVogue/apply_factor.py — same math, just packaged as one console script
that reads DV_NETWORK_PKL by default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from deepVogue import _paths


@click.group()
def cli() -> None:
    """SeFa-style factor edits."""


@cli.command("discover")
@click.option("--network", "network_pkl", type=click.Path(exists=True), default=None,
              help="defaults to $DV_NETWORK_PKL")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="defaults to $DV_RUN_DIR/factors.pt")
def discover_cmd(network_pkl: Optional[str], out: Optional[Path]) -> None:
    """Run closed-form factorization → factors.pt."""
    p = _paths.resolve()
    pkl = network_pkl or (str(p.network_pkl) if p.network_pkl else None)
    if pkl is None:
        raise click.ClickException("set --network or $DV_NETWORK_PKL")
    out = out or (p.run_dir / "factors.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "deepVogue.closed_form_factorization",
           "--ckpt", pkl, "--out", str(out)]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    click.echo(f"✓ {out}")


@cli.command("apply")
@click.option("--network", "network_pkl", type=click.Path(exists=True), default=None)
@click.option("--factors", required=True, type=click.Path(exists=True))
@click.option("--index", "-i", type=int, required=True)
@click.option("--degree", "-d", type=float, default=5.0, show_default=True)
@click.option("--seed", type=int, default=0)
@click.option("--out", type=click.Path(path_type=Path), default=Path("factor_edit.png"))
def apply_cmd(network_pkl: Optional[str], factors: str, index: int, degree: float,
              seed: int, out: Path) -> None:
    """Render a single factor edit (delegates to apply_factor.py)."""
    p = _paths.resolve()
    pkl = network_pkl or (str(p.network_pkl) if p.network_pkl else None)
    if pkl is None:
        raise click.ClickException("set --network or $DV_NETWORK_PKL")
    cmd = [sys.executable, "-m", "deepVogue.apply_factor",
           "--ckpt", pkl, "--factor", factors,
           "--index", str(index), "--degree", str(degree),
           "--seed", str(seed), "--out", str(out)]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
