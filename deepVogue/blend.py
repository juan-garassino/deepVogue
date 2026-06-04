"""Drive-aware wrapper around deepVogue/blend_models.py.

Blend two trained StyleGANs at a given resolution layer to produce a hybrid
generator — the "two-corpus aesthetic" move (e.g. tarot ⨯ film frames).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.option("--low-res-pkl", required=True, type=click.Path(exists=True),
              help="checkpoint providing low-resolution layers (overall composition)")
@click.option("--hi-res-pkl", required=True, type=click.Path(exists=True),
              help="checkpoint providing high-resolution layers (texture / detail)")
@click.option("--resolution", type=int, default=32, show_default=True,
              help="layer resolution at which the two networks are spliced")
@click.option("--out", type=click.Path(path_type=Path), default=Path("blended.pkl"))
def main(low_res_pkl: str, hi_res_pkl: str, resolution: int, out: Path) -> None:
    """Splice two checkpoints into a single blended generator."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "deepVogue.blend_models",
           "--low-res-pkl", low_res_pkl,
           "--hi-res-pkl", hi_res_pkl,
           "--resolution", str(resolution),
           "--output-path", str(out)]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    click.echo(f"✓ {out}")


if __name__ == "__main__":
    main()
