"""Latent-cinema orchestration helpers as a CLI.

Holds the bits of the pipeline that are too multi-step for a single Make recipe
but not big enough to deserve their own top-level script:

  deepvogue-cinema project-frames   # batch-project anchors out of dataset.zip
  deepvogue-cinema eval-fid         # print the FID curve for $DV_RUN_DIR
  deepvogue-cinema latest-pkl       # print latest network-snapshot-*.pkl
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import click
import numpy as np

from deepVogue import _paths


@click.group()
def cli() -> None:
    """Latent-cinema orchestration."""


@cli.command("project-frames")
@click.option("--stride", type=int, default=4, show_default=True,
              help="project every Nth anchor")
@click.option("--num-steps", type=int, default=500, show_default=True)
@click.option("--projector", "projector_kind", type=click.Choice(["lpips", "vgg"]),
              default="lpips", show_default=True,
              help="lpips = deepVogue.projector (fast); vgg = pbaylies_projector (slower, finer)")
def project_frames_cmd(stride: int, num_steps: int, projector_kind: str) -> None:
    """For each Nth frame in frames_index.json, project into DV_ANCHORS_DIR.

    Loads the generator **once** in-process (vs the old per-frame subprocess
    that re-loaded the pkl every iteration). Resume is automatic — frames whose
    `projected_w.npz` already exists are skipped.
    """
    import torch
    from PIL import Image
    from deepVogue import legacy, neuronal_network_utils
    if projector_kind == "lpips":
        from deepVogue.projector import project as project_fn
    else:
        from deepVogue.pbaylies_projector import project as project_fn

    p = _paths.resolve(require=("network_pkl",))
    idx_path = Path(p.dataset_dir) / "frames_index.json"
    if not idx_path.exists():
        raise click.ClickException(f"missing {idx_path}")
    idx = json.loads(idx_path.read_text())
    Path(p.anchors_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"[cinema] loading {p.network_pkl} on {device}  (once)")
    with neuronal_network_utils.util.open_url(str(p.network_pkl)) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)
    res = int(G.img_resolution)

    with tempfile.TemporaryDirectory(prefix="dv_proj_") as tmp:
        tmp_root = Path(tmp)
        with zipfile.ZipFile(Path(p.dataset_dir) / "dataset.zip") as zf:
            zf.extractall(tmp_root)

        n_done = 0
        n_skip = 0
        total_planned = sum(1 for i, _ in enumerate(idx["frames"]) if i % stride == 0)
        for i, frame in enumerate(idx["frames"]):
            if i % stride:
                continue
            outdir = Path(p.anchors_dir) / f"{frame['kept_index']:08d}"
            out_npz = outdir / "projected_w.npz"
            if out_npz.exists():
                n_skip += 1
                continue
            target = next(tmp_root.rglob(frame["filename"]), None)
            if target is None:
                click.echo(f"[skip] missing target for {frame['filename']}")
                continue

            img = Image.open(target).convert("RGB").resize((res, res), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)   # (C,H,W)
            t = torch.tensor(arr, dtype=torch.float32, device=device)

            if projector_kind == "lpips":
                w_traj = project_fn(G, t, num_steps=num_steps, device=device, verbose=False)
            else:
                # pbaylies signature is different — pass minimal kwargs and let defaults run
                w_traj = project_fn(G, t, None, num_steps=num_steps, device=device)
            w_final = w_traj[-1:].detach().cpu().numpy()       # (1, num_ws, w_dim)

            outdir.mkdir(parents=True, exist_ok=True)
            np.savez(out_npz, w=w_final)
            n_done += 1
            click.echo(f"  [{n_done + n_skip}/{total_planned}] {frame['filename']} → {out_npz.name}")

    click.echo(f"✓ projected {n_done} new anchors ({n_skip} skipped) into {p.anchors_dir}")


@cli.command("eval-fid")
def eval_fid_cmd() -> None:
    """Print the FID curve from $DV_RUN_DIR/metric-fid50k_full.jsonl."""
    p = _paths.resolve()
    log = next(Path(p.run_dir).rglob("metric-fid50k_full.jsonl"), None)
    if log is None:
        raise click.ClickException(f"no FID log under {p.run_dir}")
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    for r in rows:
        click.echo(f"{r['snapshot_pkl']:<40} fid={r['results']['fid50k_full']:.3f}")


@cli.command("latest-pkl")
def latest_pkl_cmd() -> None:
    """Print the path to the latest network-snapshot-*.pkl for $DV_DATASET_NAME."""
    latest = _paths.latest_snapshot()
    if latest is None:
        raise click.ClickException(f"no snapshots for DV_DATASET_NAME={_paths.resolve().dataset_name}")
    click.echo(str(latest))


if __name__ == "__main__":
    cli()
