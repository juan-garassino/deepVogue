"""CLI: register a trained checkpoint in `models.yaml`.

  python -m deepVogue.serve.register --id tarot_v1 [--pkl <path>] [--kind stills]

Defaults `--pkl` to the latest network-snapshot-*.pkl found by
`deepVogue._paths.latest_snapshot()` for the current `DV_DATASET_NAME`.
"""

from __future__ import annotations

import click

from deepVogue._paths import latest_snapshot, resolve
from .registry import Registry
from .schemas import ModelEntry


@click.command()
@click.option("--id", "model_id", required=True, help="Registry id (e.g. tarot_v1)")
@click.option("--pkl", default=None, help="Path to network-snapshot-*.pkl; defaults to latest for $DV_DATASET_NAME")
@click.option("--kind", "dataset_kind", default="stills",
              type=click.Choice(["stills", "frames"]), show_default=True)
@click.option("--trunc", "default_trunc", default=0.7, type=float, show_default=True)
@click.option("--backbone", default="sg3-t", show_default=True)
@click.option("--anchors-dir", default=None)
@click.option("--factors", default=None, help="Path to factors.pt (optional)")
@click.option("--walks-dir", default=None,
              help="Override; default = <unscoped DV_WALKS_DIR>/<id>")
def main(model_id: str, pkl, dataset_kind: str, default_trunc: float,
         backbone: str, anchors_dir, factors, walks_dir) -> None:
    if pkl is None:
        latest = latest_snapshot()
        if latest is None:
            ds = resolve().dataset_name
            raise click.ClickException(
                f"no snapshot found for DV_DATASET_NAME={ds}; pass --pkl explicitly")
        pkl = str(latest)
    entry = ModelEntry(
        id=model_id,
        backbone=backbone,
        pkl=pkl,
        dataset_kind=dataset_kind,
        default_trunc=default_trunc,
        anchors_dir=anchors_dir,
        factors=factors,
        walks_dir=walks_dir,
    )
    reg = Registry()
    reg.append_entry(entry)
    click.echo(f"✓ registered {model_id} → {pkl}")
    click.echo(f"  yaml: {reg.path}")


if __name__ == "__main__":
    main()
