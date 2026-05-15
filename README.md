# deepVogue

Latent-cinema engine built on **StyleGAN3-t** (NVIDIA's translation-equivariant
generator), with the latent-walk / projection / SeFa / network-blending tooling
from the SG2-ADA + Schultz lineage ported on top. Designed for Colab + Drive:
all datasets and outputs live on Drive, the local checkout is for code only.

## What you get

- Multi-dataset training (`DV_DATASET_NAME=tarot` / `film` / …) — Drive layout
  keeps every run reproducible and resumable across Colab session expiry.
- **Inline preprocessing**: ffmpeg-based movie-frame extraction + perceptual-hash
  dedup; procedural augmentation for tiny stills sets (rotate, crop,
  hue/sat/val jitter, gold-foil overlay, aged-paper, edge wear) — no external
  preprocessing package needed.
- **Latent cinema renderer**: project a sparse stride of frames into W+ →
  Catmull-Rom (default) / slerp / bezier / linear interpolation between
  anchors → mp4 in Drive.
- **FastAPI inference server** + **Telegram bot** running inside the same Colab
  notebook for on-demand generation and walks.

## Install

Local CPU box (code authoring + small dry-runs):
```bash
pip install -e .
```

Colab GPU runtime:
```bash
git clone --depth 1 https://github.com/JuanGarassino/deepVogue /content/deepVogue
cd /content/deepVogue
make colab-install              # pip install -e . + apt-get ffmpeg
make install-serve              # FastAPI + Telegram deps (only if running the bot)
```

## Drive layout

```
/MyDrive/deepVogue/
  data/{tarot,film}/...                 # raw stills or movie files
  datasets/{tarot,film}/dataset.zip     # prepared SG3-format zips
                       /frames_index.json   # film only
  runs/{tarot,film}/<run_id>/network-snapshot-*.pkl
  anchors/{film}/<frame_id>/projected_w.npz
  walks/{model_id}/{walk_id}.mp4
  models.yaml                           # FastAPI registry
```

Point your env vars at the Drive root and set `DV_DATASET_NAME=<name>` — every
path is suffixed automatically. See `CLAUDE.md` for the full env-var table.

## Quickstart on Colab

```bash
export DV_DATA_DIR=/content/drive/MyDrive/deepVogue/data
export DV_DATASET_DIR=/content/drive/MyDrive/deepVogue/datasets
export DV_RUN_DIR=/content/runs                 # local fast disk
export DV_DRIVE_SYNC=/content/drive/MyDrive/deepVogue/runs
export DV_ANCHORS_DIR=/content/drive/MyDrive/deepVogue/anchors
export DV_WALKS_DIR=/content/drive/MyDrive/deepVogue/walks
export DV_DATASET_NAME=tarot

# 1. pull NVIDIA SG3-t pretrained checkpoints (for fine-tune runs)
make download-pretrained

# 2. prep tarot images with procedural augmentation
make prepare-stills DV_RES=512 DV_AUGMENT=1 DV_MAX_AUG=8000

# 2b. (optional) eyeball the augmenter before kicking 24h of training
make preview-augment

# 3. train SG3-t (auto-resumes if interrupted; gamma rule: 0.0002 * res^2 / batch)
make train DV_CFG=stylegan3-t DV_KIMG=5000 DV_BATCH=32 DV_GAMMA=2
# Colab dies mid-run? Just:
make resume

# 4. register the trained checkpoint so the bot can find it
make register MODEL_ID=tarot_v1

# 5. serve + chat
make colab-serve DV_TG_TOKEN=<botfather-token> DV_TG_ALLOWLIST=<your-tg-uid>
```

## Latent cinema for a film

```bash
export DV_DATASET_NAME=film
make prepare-frames DV_RES=512 DV_FPS=1
make train DV_CFG=stylegan3-t DV_GAMMA=2 DV_NETWORK_PKL=/path/to/pretrained/stylegan3-t-ffhqu-256x256.pkl
make register MODEL_ID=film_pretrained KIND=frames
make film FILM_ID=opening_scene DV_PROJ_STRIDE=4 DV_PROJ_STEPS=500
```

## Tests

```bash
make test
```

## License

StyleGAN3 and StyleGAN2-ADA code under NVIDIA's source-code license; everything
else MIT.
