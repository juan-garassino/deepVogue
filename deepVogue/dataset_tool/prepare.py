"""Inline preprocessing — stills + movie frames → StyleGAN-format dataset.zip.

Subcommands:
  stills  — fashion / tarot / arbitrary still-image folder → dataset.zip
  frames  — folder of movie files → dataset.zip + frames_index.json (latent cinema anchor map)

Frame extraction is inline via ffmpeg (subprocess). Deduplication uses an
8x8 average-hash + Hamming distance. The final zip is produced by NVIDIA's
existing convert_dataset (deepVogue/dataset_tool.py), which we never reimplement.

Drive paths come from DV_* env vars resolved by deepVogue._paths. Override on
the CLI with --source / --out when needed.

Design lineage: the dataPalette repo at
/Users/juan-garassino/Code/005-products/004-creative-tools/004-dataPalette
inspired the structure but is *not* a runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click

from deepVogue import _paths

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _ffmpeg_extract(video: Path, out_dir: Path, fps: float, resolution: int) -> None:
    """Extract frames from `video` at `fps`, scale+center-crop to `resolution`²."""
    if shutil.which("ffmpeg") is None:
        raise click.ClickException("ffmpeg not found on PATH. Install it (Colab: !apt-get install -y ffmpeg).")
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),-1,{resolution})':'if(gt(iw,ih),{resolution},-1)',"
        f"crop={resolution}:{resolution}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vf", vf,
        "-q:v", "2",
        str(out_dir / "frame_%08d.png"),
    ]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ahash(path: Path, size: int = 8) -> str:
    """8x8 average-hash for near-duplicate frame removal (long static shots)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((size, size))
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p >= avg else "0" for p in px)
    return f"{int(bits, 2):0{size * size // 4}x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _list_videos(src: Path) -> list[Path]:
    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        return [src]
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _run_convert_dataset(source: Path, dest: Path, resolution: int) -> None:
    """Invoke NVIDIA's convert_dataset to emit the StyleGAN zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "deepVogue.dataset_tool",
        "--source", str(source),
        "--dest", str(dest),
        "--width", str(resolution),
        "--height", str(resolution),
        "--transform", "center-crop",
        "--resize-filter", "lanczos",
    ]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@click.group()
def cli() -> None:
    """Prepare StyleGAN training zips from raw stills or movie frames."""


@cli.command("stills")
@click.option("--source", type=click.Path(path_type=Path), default=None,
              help="Image folder (defaults to $DV_DATA_DIR).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Dataset directory (defaults to $DV_DATASET_DIR).")
@click.option("--resolution", "-r", type=int, default=512, show_default=True)
@click.option("--max-images", type=int, default=None)
@click.option("--augment-procedural", is_flag=True, default=False,
              help="Pre-expand a tiny set (e.g. tarot 78x2) into max-augmented variants "
                   "using rotate / crop / hsv / foil / aged-paper / edge-wear before zipping.")
@click.option("--max-augmented", type=int, default=10000, show_default=True,
              help="Target image count when --augment-procedural is set.")
@click.option("--aug-seed", type=int, default=0, show_default=True)
def stills_cmd(source: Optional[Path], out: Optional[Path], resolution: int,
               max_images: Optional[int], augment_procedural: bool,
               max_augmented: int, aug_seed: int) -> None:
    """Pack a folder of still images into dataset.zip.

    When ``--augment-procedural`` is passed, the originals are first expanded
    into ``--max-augmented`` images (originals kept verbatim, the rest are
    randomly composed augmentations) and the zip is built from that.
    """
    p = _paths.resolve()
    src = source or p.data_dir
    dst_dir = out or p.dataset_dir
    if not src.exists():
        raise click.ClickException(f"source not found: {src}")
    zip_path = dst_dir / "dataset.zip"

    if augment_procedural:
        from deepVogue.dataset_tool.tarot_augment import expand_folder
        with tempfile.TemporaryDirectory(prefix="dv_aug_") as tmp:
            tmp_dir = Path(tmp)
            kept = expand_folder(src, tmp_dir, target_count=max_augmented,
                                 resolution=resolution, seed=aug_seed)
            click.echo(f"[augment] expanded {len(list(src.rglob('*')))} sources → {kept} images")
            _run_convert_dataset(tmp_dir, zip_path, resolution)
    else:
        _run_convert_dataset(src, zip_path, resolution)

    scanned = sum(1 for _ in src.rglob('*') if _.suffix.lower() in IMAGE_EXTS)
    click.echo(f"✓ {zip_path}  ({scanned} images scanned)")


@cli.command("frames")
@click.option("--source", type=click.Path(path_type=Path), default=None,
              help="Movie file or folder of movie files (defaults to $DV_DATA_DIR).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Dataset directory (defaults to $DV_DATASET_DIR).")
@click.option("--resolution", "-r", type=int, default=512, show_default=True)
@click.option("--fps", type=float, default=1.0, show_default=True,
              help="Frames per second to sample.")
@click.option("--dedupe-threshold", type=int, default=4, show_default=True,
              help="Max Hamming distance (0-64) for treating two frames as duplicates.")
@click.option("--max-frames", type=int, default=None)
def frames_cmd(source: Optional[Path], out: Optional[Path], resolution: int,
               fps: float, dedupe_threshold: int, max_frames: Optional[int]) -> None:
    """Extract frames from movies, dedupe, pack into dataset.zip + frames_index.json."""
    p = _paths.resolve()
    src = source or p.data_dir
    dst_dir = out or p.dataset_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(src)
    if not videos:
        raise click.ClickException(f"no videos under {src}")

    with tempfile.TemporaryDirectory(prefix="dv_frames_") as tmp:
        tmp_root = Path(tmp)
        index: list[dict] = []
        kept = 0
        last_hashes: list[str] = []
        for video in videos:
            video_id = video.stem
            extract_dir = tmp_root / video_id
            click.echo(f"[extract] {video.name} → fps={fps} size={resolution}")
            _ffmpeg_extract(video, extract_dir, fps=fps, resolution=resolution)
            for frame_path in sorted(extract_dir.iterdir()):
                if max_frames is not None and kept >= max_frames:
                    break
                h = _ahash(frame_path)
                if any(_hamming(h, prev) <= dedupe_threshold for prev in last_hashes[-3:]):
                    frame_path.unlink()
                    continue
                last_hashes.append(h)
                seq = int(frame_path.stem.split("_")[-1])
                timecode = seq / fps
                index.append({
                    "video_id": video_id,
                    "source_frame": seq,
                    "timecode_s": round(timecode, 3),
                    "kept_index": kept,
                    "filename": f"img{kept:08d}.png",
                })
                target = tmp_root / f"img{kept:08d}.png"
                shutil.move(str(frame_path), str(target))
                kept += 1
            shutil.rmtree(extract_dir, ignore_errors=True)

        click.echo(f"[dedupe] kept {kept} frames")
        zip_path = dst_dir / "dataset.zip"
        _run_convert_dataset(tmp_root, zip_path, resolution)

    index_path = dst_dir / "frames_index.json"
    index_path.write_text(json.dumps({"fps": fps, "resolution": resolution,
                                      "frames": index}, indent=2))
    click.echo(f"✓ {zip_path}\n✓ {index_path}  ({kept} anchors)")


def main() -> None:
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
