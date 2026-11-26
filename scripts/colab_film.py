"""Colab per-film pipeline: train a film to a kimg target and render latent-cinema cuts.

Everything lives on Drive, so it is safe to re-run on a fresh runtime: training
resumes from the newest Drive snapshot (cumulative kimg across runs) and cuts
whose mp4 already exists are skipped. Launch it detached from one notebook cell
and the notebook (and the Colab MCP link) can go away while the runtime lives.

    python scripts/colab_film.py --train vincent256 \
        --cut spirited256:80@48 --cut vincent256:100@48 --cut vincent256:250@24

A cut ``NAME:KIMG@STRIDE`` waits until NAME has a snapshot >= KIMG on Drive
(possibly written by another notebook), projects every STRIDE-th frame into W
with that snapshot, and renders the walk to walks/NAME/NAME_k<KIMG>.mp4.
Cuts run on the same GPU as training, one at a time.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("colab_film")

D = Path(os.environ.get("DV_DRIVE_ROOT", "/content/drive/MyDrive/deepVogue"))
FFHQ = D / "pretrained/stylegan3-t-ffhqu-256x256.pkl"
LOCAL = Path("/content/film")
SNAP_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def snapshots(name: str) -> list[tuple[int, Path]]:
    """(cumulative kimg, path) for every settled snapshot of ``name`` on Drive.

    train.py restarts kimg at 0 on every resume, so each run is offset by the
    max kimg of the runs before it.
    """
    root = D / "runs" / name
    runs = sorted((d for d in root.glob("*") if d.is_dir() and d.name[:5].isdigit()),
                  key=lambda d: int(d.name[:5])) if root.exists() else []
    out, offset = [], 0
    for d in runs:
        ks = sorted((int(SNAP_RE.search(p.name).group(1)), p) for p in d.glob("network-snapshot-*.pkl"))
        out += [(offset + k, p) for k, p in ks if time.time() - p.stat().st_mtime > 120]
        if ks:
            offset += ks[-1][0]
    return out


def local_dataset(name: str) -> Path:
    """Copy dataset.zip + frames_index.json to local disk once; returns the dir."""
    dst = LOCAL / "datasets" / name
    if not (dst / "frames_index.json").exists():
        src = D / "datasets" / name
        while not ((src / "dataset.zip").exists() and (src / "frames_index.json").exists()):
            log.info("%s: waiting for dataset on Drive", name)
            time.sleep(120)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "dataset.zip", dst / "dataset.zip")
        shutil.copy(src / "frames_index.json", dst / "frames_index.json")
    return dst


def start_training(name: str, target: int, batch: int, gamma: float):
    snaps = snapshots(name)
    done = snaps[-1][0] if snaps else 0
    if done >= target:
        log.info("%s: already at %d >= %d kimg", name, done, target)
        return None, None
    ds = local_dataset(name)
    run = LOCAL / "runs" / name
    # Placeholder dirs so train.py's run id continues after Drive's (never overwrites a run).
    for d in (D / "runs" / name).glob("*"):
        if d.is_dir():
            (run / d.name).mkdir(parents=True, exist_ok=True)
    resume = snaps[-1][1] if snaps else FFHQ
    cmd = [sys.executable, "-m", "deepVogue.train", "--outdir", str(run), "--cfg", "stylegan3-t",
           "--data", str(ds / "dataset.zip"), "--gpus", "1", "--batch", str(batch), "--gamma", str(gamma),
           "--cbase", "16384", "--mirror", "1", "--aug", "ada", "--kimg", str(target - done),
           "--snap", "5", "--metrics", "none", "--workers", "6", "--resume", str(resume)]
    logf = open(LOCAL / f"train_{name}.log", "a")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    from deepVogue._drive_sync import DriveSync
    sync = DriveSync(run, D / "runs" / name, 60).start()
    log.info("%s: training %d → %d kimg from %s (pid %d)", name, done, target, resume, proc.pid)
    return proc, sync


def walk(pkl: Path, anchors: Path, ds: Path, out_mp4: Path, fps_per_seg: int, env: dict, logf) -> None:
    tmp_mp4 = LOCAL / out_mp4.name
    subprocess.run([sys.executable, "-m", "deepVogue.walk", "--network", str(pkl), "--anchors", str(anchors),
                    "--order", str(ds / "frames_index.json"), "--out", str(tmp_mp4), "--interp", "cubic",
                    "--noise-mode", "const", "--fps", "24", "--frames-per-segment", str(fps_per_seg)],
                   env=env, stdout=logf, stderr=subprocess.STDOUT, check=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_mp4, out_mp4)


def render_cut(name: str, kimg: int, stride: int, steps: int, fps_per_seg: int, ft_steps: int) -> bool:
    """Project + walk (+ film-tune + walk) for the first snapshot >= kimg. False if not ready yet."""
    out_mp4 = D / "walks" / name / f"{name}_k{kimg:04d}.mp4"
    ft_mp4 = out_mp4.with_name(out_mp4.stem + "_ft.mp4")
    if out_mp4.exists() and (ft_steps <= 0 or ft_mp4.exists()):
        return True
    ready = [(k, p) for k, p in snapshots(name) if k >= kimg]
    if not ready:
        return False
    k, snap = ready[0]
    ds = local_dataset(name)
    pkl = LOCAL / "pkls" / f"{name}_k{k:04d}.pkl"
    if not pkl.exists():
        pkl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(snap, pkl)
    anchors = D / "anchors" / name / f"k{kimg:04d}_s{stride}"
    env = {k_: v for k_, v in os.environ.items() if k_ != "DV_DATASET_NAME"}
    env.update(DV_DATASET_DIR=str(ds), DV_NETWORK_PKL=str(pkl), DV_ANCHORS_DIR=str(anchors))
    logf = open(LOCAL / f"cut_{name}_k{kimg:04d}.log", "a")
    if not out_mp4.exists():
        log.info("%s: cut k%d (snapshot %d kimg, stride %d) → projecting", name, kimg, k, stride)
        subprocess.run([sys.executable, "-m", "deepVogue.cinema", "project-frames",
                        "--stride", str(stride), "--num-steps", str(steps)],
                       env=env, stdout=logf, stderr=subprocess.STDOUT, check=True)
        log.info("%s: cut k%d → rendering walk", name, kimg)
        walk(pkl, anchors, ds, out_mp4, fps_per_seg, env, logf)
        log.info("%s: cut k%d done → %s", name, kimg, out_mp4)
    if ft_steps > 0 and not ft_mp4.exists():
        ft_pkl = LOCAL / "pkls" / f"{name}_k{k:04d}_ft.pkl"
        log.info("%s: cut k%d → film-tuning %d steps", name, kimg, ft_steps)
        subprocess.run([sys.executable, "-m", "deepVogue.film_tune", "--network", str(pkl), "--anchors", str(anchors),
                        "--dataset-dir", str(ds), "--out", str(ft_pkl), "--steps", str(ft_steps)],
                       env=env, stdout=logf, stderr=subprocess.STDOUT, check=True)
        shutil.copy(ft_pkl.with_suffix(".png"), ft_mp4.with_suffix(".png"))
        shutil.copy(ft_pkl, D / "runs" / name / ft_pkl.name)
        walk(ft_pkl, ft_pkl.with_suffix(".anchors"), ds, ft_mp4, fps_per_seg, env, logf)
        log.info("%s: cut k%d film-tuned → %s", name, kimg, ft_mp4)
    return True


def parse_cut(spec: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"([\w-]+):(\d+)@(\d+)", spec)
    if not m:
        raise argparse.ArgumentTypeError(f"cut must be NAME:KIMG@STRIDE, got {spec!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", help="film to train in this runtime (dataset name)")
    ap.add_argument("--target", type=int, default=512, help="cumulative kimg target")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--cut", type=parse_cut, action="append", default=[], help="NAME:KIMG@STRIDE")
    ap.add_argument("--steps", type=int, default=300, help="projection steps per anchor")
    ap.add_argument("--fps-per-seg", type=int, default=24, help="walk frames between anchors")
    ap.add_argument("--film-tune-steps", type=int, default=0,
                    help="also film-tune G on each cut's anchors (deepVogue.film_tune) and render <cut>_ft.mp4")
    args = ap.parse_args()

    LOCAL.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOCAL / "colab_film.log")])
    proc, sync = start_training(args.train, args.target, args.batch, args.gamma) if args.train else (None, None)
    pending = sorted(args.cut, key=lambda c: c[1])
    while pending or (proc and proc.poll() is None):
        for cut in list(pending):
            try:
                if render_cut(*cut, steps=args.steps, fps_per_seg=args.fps_per_seg, ft_steps=args.film_tune_steps):
                    pending.remove(cut)
            except subprocess.CalledProcessError:
                log.exception("cut %s failed (see cut log); dropping it", cut)
                pending.remove(cut)
        time.sleep(120)
    if proc:
        log.info("training exited with %s", proc.returncode)
    if sync:
        sync.stop()


if __name__ == "__main__":
    main()
