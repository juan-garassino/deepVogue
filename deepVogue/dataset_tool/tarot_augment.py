"""Procedural augmentation for the 78x2 tarot dataset.

The base set is tiny (78 Rider-Waite cards × 2 decks ≈ 156 images), so SG3-t
needs many more samples even with ADA enabled. This module pre-generates a
larger augmented set BEFORE NVIDIA's ``convert_dataset`` packs the zip, so the
on-disk dataset is already inflated to ``max_augmented`` images and the
training loop sees a varied corpus from kimg 0.

Augmentations applied per source image (composed randomly):
  - rotate ±5°
  - random crop + pad back to canvas
  - hue / saturation / value jitter
  - simulated gold-foil edge overlay (alpha-blended radial gradient)
  - aged-paper overlay (sepia tint + low-freq noise)
  - edge wear (multiplicative frayed mask)

All ops are inline numpy + PIL — no external deps beyond what
``requirements.txt`` already ships. The dataPalette repo at
``/Users/juan-garassino/Code/005-products/004-creative-tools/004-dataPalette``
is **reference only**, not a runtime dependency.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _rotate(img: Image.Image, max_deg: float = 5.0) -> Image.Image:
    deg = random.uniform(-max_deg, max_deg)
    return img.rotate(deg, resample=Image.BICUBIC, expand=False,
                      fillcolor=(0, 0, 0))


def _crop_pad(img: Image.Image, max_frac: float = 0.08) -> Image.Image:
    w, h = img.size
    dx = int(random.uniform(0, max_frac) * w)
    dy = int(random.uniform(0, max_frac) * h)
    left, top = random.randint(0, dx), random.randint(0, dy)
    cropped = img.crop((left, top, w - (dx - left), h - (dy - top)))
    return cropped.resize((w, h), Image.BICUBIC)


def _hsv_jitter(img: Image.Image, hue: float = 0.04, sat: float = 0.20,
                val: float = 0.15) -> Image.Image:
    arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    arr[..., 0] = (arr[..., 0] + random.uniform(-hue, hue)) % 1.0
    arr[..., 1] = np.clip(arr[..., 1] * (1 + random.uniform(-sat, sat)), 0, 1)
    arr[..., 2] = np.clip(arr[..., 2] * (1 + random.uniform(-val, val)), 0, 1)
    out = (arr * 255).astype(np.uint8)
    return Image.fromarray(out, mode="HSV").convert("RGB")


def _gold_foil(img: Image.Image, strength: float = 0.18) -> Image.Image:
    """Alpha-blend a warm radial gradient over the card."""
    w, h = img.size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * random.uniform(0.3, 0.7), h * random.uniform(0.3, 0.7)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = 1.0 - np.clip(r / (0.6 * max(w, h)), 0, 1)
    foil = np.stack([
        255 * (0.95 + 0.05 * r),
        220 * (0.85 + 0.15 * r),
        140 * (0.65 + 0.35 * r),
    ], axis=-1).astype(np.uint8)
    return Image.blend(img.convert("RGB"), Image.fromarray(foil), strength)


def _aged_paper(img: Image.Image, strength: float = 0.22) -> Image.Image:
    """Sepia tint + soft low-freq noise to mimic worn cardboard."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    sepia = np.stack([
        arr[..., 0] * 0.393 + arr[..., 1] * 0.769 + arr[..., 2] * 0.189,
        arr[..., 0] * 0.349 + arr[..., 1] * 0.686 + arr[..., 2] * 0.168,
        arr[..., 0] * 0.272 + arr[..., 1] * 0.534 + arr[..., 2] * 0.131,
    ], axis=-1)
    sepia = np.clip(sepia, 0, 255)
    # low-freq noise
    h, w = arr.shape[:2]
    noise_small = np.random.RandomState(random.randint(0, 1 << 31)).rand(
        max(8, h // 32), max(8, w // 32)
    )
    noise = np.array(
        Image.fromarray((noise_small * 255).astype(np.uint8))
        .resize((w, h), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    blended = arr * (1 - strength) + sepia * strength
    blended = blended * (0.92 + 0.08 * noise[..., None])
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _edge_wear(img: Image.Image, strength: float = 0.35) -> Image.Image:
    """Multiplicative mask darkening corners + frayed irregular edges."""
    w, h = img.size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # corner vignette
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    vign = 1.0 - np.clip(np.sqrt(nx ** 2 + ny ** 2) - 0.8, 0, 1) ** 2 * strength
    # frayed irregular edge: low-freq noise gated near borders
    rs = np.random.RandomState(random.randint(0, 1 << 31))
    fray_small = rs.rand(max(8, h // 16), max(8, w // 16))
    fray = np.array(
        Image.fromarray((fray_small * 255).astype(np.uint8))
        .resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    border_dist = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / min(w, h)
    border_mask = 1.0 - np.clip(border_dist / 0.05, 0, 1)
    edge = 1.0 - border_mask * (0.4 + 0.6 * fray) * strength
    mask = (vign * edge).clip(0, 1)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return Image.fromarray((arr * mask[..., None]).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

_OPS = [
    ("rotate", _rotate, 0.65),
    ("crop_pad", _crop_pad, 0.55),
    ("hsv", _hsv_jitter, 0.70),
    ("gold_foil", _gold_foil, 0.20),
    ("aged", _aged_paper, 0.35),
    ("edge_wear", _edge_wear, 0.35),
]


def augment_once(img: Image.Image) -> Image.Image:
    """Apply a random subset of augmentations to a single image."""
    out = img
    for _, op, prob in _OPS:
        if random.random() < prob:
            out = op(out)
    return out


def expand_folder(src_dir: Path, dst_dir: Path, target_count: int,
                  resolution: int, seed: int = 0,
                  image_exts: Iterable[str] = (".png", ".jpg", ".jpeg",
                                               ".bmp", ".webp")) -> int:
    """Pre-generate ``target_count`` augmented images from the originals in
    ``src_dir`` into ``dst_dir`` (a flat folder). Returns total files written.

    The originals are kept verbatim (no augmentation) so the model sees the
    canonical cards too. The remainder is filled with composed augmentations.
    """
    random.seed(seed)
    np.random.seed(seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    originals = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in image_exts])
    if not originals:
        raise FileNotFoundError(f"no source images under {src_dir}")

    # copy originals first
    kept = 0
    for src in originals:
        img = Image.open(src).convert("RGB")
        if img.size != (resolution, resolution):
            img = ImageOps.fit(img, (resolution, resolution), Image.BICUBIC)
        img.save(dst_dir / f"img{kept:08d}.png", optimize=False)
        kept += 1

    # fill with augmented variants
    while kept < target_count:
        src = random.choice(originals)
        img = Image.open(src).convert("RGB")
        if img.size != (resolution, resolution):
            img = ImageOps.fit(img, (resolution, resolution), Image.BICUBIC)
        aug = augment_once(img)
        aug.save(dst_dir / f"img{kept:08d}.png", optimize=False)
        kept += 1

    return kept
