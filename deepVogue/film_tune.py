"""Film-tuning: overfit a trained generator to its film (multi-anchor pivotal tuning).

GAN training only ever shows G the film through D, so it learns "frames like
these", not the frames. Film-tuning closes that gap after projection: with the
anchors' w_i frozen, fine-tune G.synthesis so G(w_i) reproduces frame i
(LPIPS + L2). A locality term keeps what a walk visits *between* anchors —
lerps between temporally adjacent anchors — close to the untuned generator,
so in-betweens stay smooth instead of collapsing into artifacts.

    python -m deepVogue.film_tune --network net.pkl --anchors anchors/ \
        --dataset-dir datasets/spirited256 --out net_ft.pkl --steps 2000

Writes ``<out>.pkl`` (``{"G_ema": G}``, loadable by walk/serve) and
``<out>.png`` (rows: target / untuned / tuned for 8 anchors).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import pickle
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VGG_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"


def load_pairs(anchors_dir: Path, dataset_dir: Path, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Anchors in film order → (w [N, L, C] float32, frames [N, 3, H, W] uint8)."""
    from PIL import Image

    idx = json.loads((dataset_dir / "frames_index.json").read_text())
    ws, imgs = [], []
    with zipfile.ZipFile(dataset_dir / "dataset.zip") as zf:
        by_name = {Path(n).name: n for n in zf.namelist()}
        for frame in idx["frames"]:
            npz = anchors_dir / f"{frame['kept_index']:08d}" / "projected_w.npz"
            if not npz.exists():
                continue
            ws.append(np.load(npz)["w"][0].astype(np.float32))
            img = Image.open(io.BytesIO(zf.read(by_name[Path(frame["filename"]).name])))
            img = img.convert("RGB").resize((res, res), Image.BICUBIC)
            imgs.append(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    if not ws:
        raise SystemExit(f"no anchors under {anchors_dir}")
    return np.stack(ws), np.stack(imgs)


def tune(G, ws: np.ndarray, imgs: np.ndarray, *, steps: int, batch: int, lr: float,
         l2_weight: float, locality_weight: float, device):
    import torch
    import torch.nn.functional as F
    from deepVogue.neuronal_network_utils import util

    G_ref = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    G.synthesis.requires_grad_(True)
    with util.open_url(VGG_URL) as f:
        vgg = torch.jit.load(f).eval().to(device)

    def lpips(a, b):  # a, b in [0, 255]
        fa = vgg(a, resize_images=False, return_lpips=True)
        fb = vgg(b, resize_images=False, return_lpips=True)
        return (fa - fb).square().sum(dim=1)

    def synth(net, w):
        return (net.synthesis(w, noise_mode="const") + 1) * 127.5

    W = torch.from_numpy(ws).to(device)
    X = torch.from_numpy(imgs).to(device)  # uint8 on device; cast per batch
    n = len(W)
    opt = torch.optim.Adam(G.synthesis.parameters(), lr=lr, betas=(0.9, 0.999))
    rng = np.random.default_rng(0)
    for step in range(1, steps + 1):
        i = torch.from_numpy(rng.integers(0, n, batch)).to(device)
        out, target = synth(G, W[i]), X[i].float()
        rec = lpips(out, target).mean()
        l2 = F.mse_loss(out / 127.5, target / 127.5)
        loss = rec + l2_weight * l2
        loc = torch.zeros((), device=device)
        if locality_weight > 0 and n > 1:
            j = torch.from_numpy(rng.integers(0, n - 1, batch)).to(device)
            t = torch.rand(batch, 1, 1, device=device)
            w_mid = torch.lerp(W[j], W[j + 1], t)
            with torch.no_grad():
                ref = synth(G_ref, w_mid)
            loc = lpips(synth(G, w_mid), ref).mean()
            loss = loss + locality_weight * loc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            log.info("step %d/%d  lpips %.4f  l2 %.4f  locality %.4f", step, steps, rec.item(), l2.item(), loc.item())
    return G_ref, G.eval().requires_grad_(False)


def preview(G_ref, G, ws: np.ndarray, imgs: np.ndarray, path: Path, device, k: int = 8) -> None:
    import torch
    from PIL import Image

    sel = np.linspace(0, len(ws) - 1, min(k, len(ws))).astype(int)
    w = torch.from_numpy(ws[sel]).to(device)
    rows = [imgs[sel]]
    with torch.no_grad():
        for net in (G_ref, G):
            img = (net.synthesis(w, noise_mode="const") + 1) * 127.5
            rows.append(img.clamp(0, 255).to(torch.uint8).cpu().numpy())
    grid = np.concatenate([np.concatenate(list(r), axis=2) for r in rows], axis=1)  # [3, 3H, kW]
    Image.fromarray(grid.transpose(1, 2, 0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True, type=Path)
    ap.add_argument("--anchors", required=True, type=Path, help="dir of <kept_index>/projected_w.npz")
    ap.add_argument("--dataset-dir", required=True, type=Path, help="dir with dataset.zip + frames_index.json")
    ap.add_argument("--out", required=True, type=Path, help="output .pkl (a .png preview is written beside it)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l2-weight", type=float, default=1.0)
    ap.add_argument("--locality-weight", type=float, default=0.5,
                    help="pull in-between (lerped) latents toward the untuned G; 0 = pure overfit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from deepVogue import _runtime

    device = _runtime.pick_device()
    G = _runtime.load_generator(args.network, device)
    ws, imgs = load_pairs(args.anchors, args.dataset_dir, int(G.img_resolution))
    log.info("film-tuning on %d anchors for %d steps", len(ws), args.steps)
    G_ref, G_ft = tune(G, ws, imgs, steps=args.steps, batch=args.batch, lr=args.lr,
                       l2_weight=args.l2_weight, locality_weight=args.locality_weight, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preview(G_ref, G_ft, ws, imgs, args.out.with_suffix(".png"), device)
    with open(args.out, "wb") as f:
        pickle.dump({"G_ema": copy.deepcopy(G_ft).cpu()}, f)
    log.info("wrote %s (+ .png preview)", args.out)


if __name__ == "__main__":
    main()
