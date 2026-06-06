# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**deepVogue** is a **StyleGAN3-t base** (NVIDIA's `stylegan3`) with the latent-cinema features from the SG2-ADA / Schultz lineage ported on top — used as the engine for a data-driven generative-art / latent-cinema project.

The training stack (`deepVogue/train.py`, `deepVogue/training/{training_loop,dataset,loss,augment,networks_stylegan2,networks_stylegan3}.py`, `deepVogue/pytorch_utils/`) is vendored from upstream NVlabs/stylegan3. The legacy SG2-ADA training files are preserved as `*_legacy.py` for reference (not imported), and `deepVogue/training/networks.py` (original SG2-ADA Generator) is kept solely so old `.pkl` checkpoints unpickle correctly through `legacy.py`. Inference/art scripts (projector, walk, factors, blend, cinema) are deepVogue-specific and being adapted to the SG3 generator (Phase 4).

### Artistic / Research Vision (project intent)

This repo is being pushed beyond pure ML training toward **data-driven generative art** in the spirit of Refik Anadol — but more science- and data-driven. Concrete directions to keep in mind when proposing changes or new tools:

- **Artistic bias from the dataset.** The aesthetic is treated as an emergent property of the training corpus; dataset curation is a first-class artistic act, not a preprocessing step.
- **Latent cinema.** Train/project a model so individual movie frames become anchor points in W/W+ space, then render the film as a continuous latent walk between those anchors (interpolation, audio-reactive modulation, factor edits). Relevant existing pieces: `projector.py`, `pbaylies_projector.py`, `notebooks/SG2-ADA-PT_AudioReactive+Pitch.ipynb`, `notebooks/Network_Blending_ADA_PT.ipynb`.
- **Refik-Anadol-style installations, more rigorous.** Favor reproducible, parameterized walks (seeded noise fields, OpenSimplex trajectories already used in `generate.py`) and expose the science: factor directions from `closed_form_factorization.py` / `apply_factor.py`, model blending from `blend_models.py`, metrics from `calc_metrics.py`.

When extending the project, prefer additions that make the latent space *navigable, recordable, and renderable as time-based media* over one-off image generation features.

## Runtime model — Colab-first, Drive-only data

**deepVogue never reads datasets from the local machine.** All training and all data live in Colab + Google Drive; the local checkout is for code authoring and small dry-runs only. Every dataset/output path is read from a `DV_*` environment variable resolved by `deepVogue/_paths.py`. No script hardcodes a path.

| Env var          | Purpose |
|------------------|---------|
| `DV_DATA_DIR`    | Preprocessing input — raw stills, or movie files |
| `DV_DATASET_DIR` | Where the prepared StyleGAN2-ADA `dataset.zip` (and `frames_index.json` for movies) lands |
| `DV_RUN_DIR`     | Training outputs (snapshots, `.pkl`, fakes, FID log) — usually `/content/runs/...` on Colab |
| `DV_DRIVE_SYNC`  | Mirror of `DV_RUN_DIR` on Drive; written to every minute by `deepVogue/_drive_sync.py::DriveSync` |
| `DV_NETWORK_PKL` | Default checkpoint for `deepvogue-{generate,project,walk,factors,blend}` |
| `DV_ANCHORS_DIR` | Per-frame projected W+ vectors (`<id>/projected_w.npz`) — input to `deepvogue-walk` |
| `DV_WALKS_DIR`   | Rendered mp4s |
| `DV_DATASET_NAME`| Optional name (`tarot`, `film`, `tarot_pretrained`, …). When set, `dataset_dir` / `run_dir` / `drive_sync` / `anchors_dir` / `walks_dir` are all suffixed with it so multiple datasets coexist under one Drive root. |
| `DV_MODELS_YAML` | Optional override for the FastAPI registry yaml; defaults to `<DV_DRIVE_SYNC parent>/models.yaml` |

**Multi-dataset layout.** Set `DV_DATASET_NAME=tarot` (or any name) and a single Drive root holds `runs/tarot/...`, `runs/film/...`, `runs/tarot_pretrained/...`, etc. `make resume` looks at the latest `network-snapshot-*.pkl` under `$DV_DRIVE_SYNC/<name>/` (fallback `$DV_RUN_DIR/<name>/`) and re-launches `make train` with `DV_NETWORK_PKL` pointed at it — survives Colab session expiry. `make latest-pkl` prints just the path. `make models-list` prints the registry yaml.

## Serving & bot

`deepVogue/serve/` is a FastAPI app and `deepVogue/bot/telegram.py` is a python-telegram-bot v20 long-poller. Both run inside the same Colab notebook via `make colab-serve`. The bot only talks to FastAPI over localhost; FastAPI owns the GPU.

| Env var          | Purpose |
|------------------|---------|
| `DV_FASTAPI_HOST`/`DV_FASTAPI_PORT` | bind address (default `0.0.0.0:8080`) |
| `DV_FASTAPI_URL` | where the bot reaches the server (default `http://127.0.0.1:8080`) |
| `DV_TG_TOKEN`    | BotFather token (required to launch the bot) |
| `DV_TG_ALLOWLIST`| comma-separated Telegram user IDs (empty = open) |
| `DV_MODELS_YAML` | registry yaml; defaults to `<DV_DRIVE_SYNC parent>/models.yaml` |
| `DV_PRETRAINED_DIR` | where `make download-pretrained` lands SG3 checkpoints |

`models.yaml` schema (list of entries):
```yaml
- id: tarot_pretrained
  backbone: sg3-t
  pkl: runs/tarot_pretrained/2026-05-15/network-snapshot-005000.pkl
  dataset_kind: stills
  default_trunc: 0.7
  factors: factors/tarot_pretrained.pt    # optional, for /factor
  anchors_dir: anchors/tarot              # optional, for /film
```

Endpoints: `GET /health`, `GET /models`, `GET /status?model=<id>`, `POST /generate {model,seed,trunc,factor_idx?,factor_amp?}`, `POST /walk {model,seeds[],steps,fps,mode,trunc?}`, `GET /films/{model_id}/{walk_id}`. Bot commands: `/start /models /gen /walk /film /factor /status`.

**Films endpoint contract.** `GET /films/{model_id}/{walk_id}` resolves to `<entry.walks_dir>/<walk_id>.mp4` if the registry entry overrides `walks_dir`, else to `<unscoped $DV_WALKS_DIR>/<model_id>/<walk_id>.mp4`. The lookup is **independent of `DV_DATASET_NAME`** so the bot can serve films from any registered dataset regardless of which dataset FastAPI was launched against.

The canonical entry point is `notebooks/deepVogue_colab.ipynb` (cells 00–07: setup → configure → prepare → train → project anchors → walk → factors → eval). Every step in the notebook is a `make` target — the notebook just sets `DV_*` env vars and calls `!make <target>`. The same commands run locally.

**Bootstrap on Colab is GitHub-clone**: `git clone --depth 1 --branch master $GITHUB_URL /content/deepVogue` followed by `make colab-install` (which `pip install -e .`s the repo and apt-installs ffmpeg). Drive holds *data and outputs only* (`/MyDrive/deepVogue/{data,datasets,runs,anchors,walks}/...`); the code is never zipped onto Drive.

## MLOps stack

Local-nano stack (docker-compose) mirrors the prod-GCP stack one-for-one. Same code, same `models.yaml`, same flows — only the artifact backend (`DV_ARTIFACT_BACKEND=s3` → MinIO vs `gcs` → GCS) and endpoint URLs change.

| Env var | Purpose |
|---|---|
| `DV_ARTIFACT_BACKEND` | `s3` (MinIO local), `gcs` (prod), `memory` (tests), `file` (default) |
| `DV_S3_ENDPOINT_URL` | MinIO URL when backend=s3 |
| `DV_MODELS_YAML` | Full URI to `models.yaml` (e.g. `gs://deepvogue-models/models.yaml`) |
| `DV_MODELS_ROOT` | Resolution root for relative `pkl:` entries (back-compat) |
| `DV_PUBLISH_TARGET` | Where `make publish` uploads (e.g. `gs://deepvogue-models`) |
| `MLFLOW_TRACKING_URI` | MLflow server URL (Cloud Run in prod) |
| `MLFLOW_TRACKING_TOKEN` | IAP id_token for programmatic access from Colab |
| `PREFECT_API_URL` | Prefect server API URL |
| `SLACK_WEBHOOK_URL` | Optional; if unset, Slack helpers no-op |

GCS bucket layout (prod): `gs://deepvogue-{models,datasets,walks,mlflow,queue}`. Artifact Registry: `us-central1-docker.pkg.dev/<project>/deepvogue/{inference,mlflow,prefect-server,prefect-worker,train}`.

New Makefile targets:
- `make nano-up` / `nano-down` / `nano-logs` — local docker-compose stack
- `make nano-smoke` — full pipeline against the local stack
- `make publish MODEL_ID=...` — Drive → GCS + `models.yaml` append + Slack
- `make gcp-setup` — runs `infra/gcp/setup.sh` + setup-sql.sh + setup-iam.sh
- `make deploy-inference` / `deploy-mlflow` / `deploy-prefect` — Cloud Run deploy

Prefect flows live in `deepVogue/orchestration/flows.py`. Each task dispatches via `get_backend(backend)`; v1 supports `backend="local"` (real `prepare`/`publish`, mock `train`/`project`/`walk`/`eval`) and `backend="runpod"` (real `train` on a RunPod GPU pod; the rest delegate to `local`). `colab` is scaffolded but raises `NotImplementedError` — manual Colab + `make publish` is the Colab path.

Colab logs to remote MLflow via `deepVogue/tracking/mlflow_helpers.py::log_training_run`. IAP auth: notebook cell sets `MLFLOW_TRACKING_TOKEN` from `gcloud auth print-identity-token --audiences=<iap-oauth-client-id>`.

### RunPod training backend

`backend="runpod"` submits a GPU pod that pulls the dataset from GCS, runs `deepVogue/train.py`, rsync-mirrors snapshots to `DV_RUN_URI`, and publishes the final checkpoint to `DV_PUBLISH_TARGET/models.yaml` — all via the same `publish_checkpoint` path as Colab.

**Auth setup (one-time):**
- `RUNPOD_API_KEY` from runpod.io/console/user/settings
- `GOOGLE_APPLICATION_CREDENTIALS_JSON=$(cat trainer-key.json)` minted by `gcloud iam service-accounts keys create trainer-key.json --iam-account=deepvogue-trainer-sa@$GCP_PROJECT.iam.gserviceaccount.com` (the SA is created by `make gcp-setup`)
- GHCR `deepvogue-train` package made public on first build (CI does this; private alternative is `runpod.create_container_registry_auth` → `create_template`)

**Pod lifecycle:** the pod's entrypoint calls RunPod's GraphQL `podTerminate` on exit (success or failure). The orchestrator polls `get_pod` and treats `TERMINATED` / `EXITED` / pod-gone as success, `FAILED` / `DEAD` as failure, and force-terminates on `RUNPOD_MAX_TRAIN_HOURS` timeout. Final guard `terminate_pod` is idempotent.

| Env var (launcher side) | Purpose |
|---|---|
| `RUNPOD_API_KEY` | Auth |
| `RUNPOD_IMAGE` | GHCR image, e.g. `ghcr.io/<owner>/deepvogue-train:latest` |
| `RUNPOD_GPU_TYPE` | GPU SKU (default `NVIDIA H100 80GB HBM3`) |
| `RUNPOD_VOLUME_GB` | Persistent volume (default 0 — outputs mirror to GCS) |
| `RUNPOD_CONTAINER_DISK_GB` | Ephemeral disk (default 100) |
| `RUNPOD_MAX_TRAIN_HOURS` | Hard timeout; force-terminate past this (default 24) |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Full SA JSON; activated on pod startup |
| `DV_DATASET_URI` | `gs://...` dataset.zip URI |
| `DV_RUN_URI` | `gs://...` snapshot mirror prefix |
| `DV_PUBLISH_TARGET` | `gs://...` root holding `models.yaml` |

Targets: `make runpod-build` / `runpod-push` build+push the train image (GHCR), `make runpod-train MODEL_ID=tarot` calls `scripts/submit_runpod_train.py` (emits JSON events to stdout), `make runpod-status POD_ID=...` / `make runpod-logs POD_ID=...` / `make runpod-terminate POD_ID=...` for ops. CI: `.github/workflows/build-train.yml` auto-builds, pushes to GHCR, makes the package public, and runs a `DV_FAKE_TRAIN=1` smoke that exercises the entrypoint without a GPU.

Per-tick MLflow logging from the pod is **deferred to v2** (IAP id-token broker needed from non-GCP infra); v1 captures the final FID in `models.yaml` via `publish`.

## Common Commands

The `Makefile` is the canonical entry point for repo housekeeping; ML workflows are run directly via the scripts in `deepVogue/`.

```bash
# install
pip install -e .                              # local / inference only
pip install -e . -r requirements-train.txt    # full training stack
make colab-install                            # what notebook cell 00 runs (pip install -e . + apt-get ffmpeg)
make colab-clone DV_GITHUB_URL=<url>          # clones repo into /content/deepVogue

# repo housekeeping
make black check_code test clean

# data prep — inline ffmpeg+PIL → StyleGAN zip
make prepare-stills DV_RES=512                # tarot / fashion folder
make prepare-frames DV_RES=512 DV_FPS=1       # movies → dataset.zip + frames_index.json

# training (Drive-aware; reads DV_RUN_DIR / DV_DRIVE_SYNC / optional DV_NETWORK_PKL)
# Gamma rule of thumb: ≈ 0.0002 * res^2 / batch  (256→0.5, 512→2, 1024→6.6)
make train DV_DATASET_NAME=tarot DV_CFG=stylegan3-t DV_KIMG=5000 DV_BATCH=32 DV_GAMMA=2
make resume DV_DATASET_NAME=tarot             # resumes from latest Drive snapshot
make latest-pkl DV_DATASET_NAME=tarot         # prints latest network-snapshot-*.pkl path
make models-list                              # prints models.yaml registry
make register MODEL_ID=tarot_v1               # appends latest snapshot to models.yaml
make download-pretrained                      # NVIDIA SG3-t pretrained pkls → DV_PRETRAINED_DIR
make preview-augment DV_DATASET_NAME=tarot    # dump 20 augmented samples for visual sanity
make sync-out                                 # one-shot mirror of DV_RUN_DIR → DV_DRIVE_SYNC

# latent cinema
make project-frames DV_PROJ_STRIDE=4 DV_PROJ_STEPS=500   # batch project anchors out of dataset.zip
make walk                                     # uses frames_index.json if available
make walk-stills                              # alphabetical anchor order
make walk-factor DV_FACTOR_IDX=7 DV_FACTOR_AMP=0.4

# Refik-style edits
make factors-discover
make blend LOW=<low.pkl> HIGH=<hi.pkl> DV_BLEND_RES=32

# eval
make eval                                     # prints FID curve from metric-fid50k_full.jsonl

# one-shot pipelines (compose the above)
make pipeline-stills                          # prepare-stills → train → walk-stills
make pipeline-frames                          # prepare-frames → train → project-frames → walk → eval

# inspect resolved env
make env

# train a model (StyleGAN2-ADA)
python deepVogue/train.py --outdir=./results --data=<path-to-zip-or-dir> --gpus=1 --cfg=auto

# generate images / latent walks
python deepVogue/generate.py --network=<pkl> --outdir=out --seeds=0-31 --trunc=0.7

# project a real image into W (anchor for latent cinema)
python deepVogue/projector.py --network=<pkl> --target=<frame.png> --outdir=out
python deepVogue/pbaylies_projector.py ...      # extended projector with W+ / VGG perceptual

# discover semantic edit directions and apply them
python deepVogue/closed_form_factorization.py --ckpt=<pkl> --out=factors.pt
python deepVogue/apply_factor.py --network=<pkl> --factors=factors.pt -i <idx> -d <degree>

# blend / mix two trained networks
python deepVogue/blend_models.py --low-res-pkl=A.pkl --hi-res-pkl=B.pkl --resolution=<N>
python deepVogue/style_mixing.py --network=<pkl> --rows=... --cols=...

# build training dataset (zips of square images)
python deepVogue/dataset_tool.py --source=<images-dir> --dest=<dataset.zip>

# evaluate a checkpoint
python deepVogue/calc_metrics.py --network=<pkl> --metrics=fid50k_full --data=<dataset.zip>
```

GPU runtime is the official NVIDIA PyTorch container — use `docker_run.sh` (wrapper around `Dockerfile`, base `nvcr.io/nvidia/pytorch:20.12-py3`). The custom CUDA ops in `deepVogue/pytorch_utils/ops/` JIT-compile on first run and require a working `nvcc`.

Run a single test: `pytest tests/test_<name>.py::<test_func>` (no tests are checked in yet beyond `tests/__init__.py` — the test target exists but is currently empty).

## Architecture

The package is the StyleGAN2-ADA-PyTorch codebase split into purpose-named subpackages. Imports inside scripts often use bare names (e.g. `import neuronal_network_utils`, `import legacy`) rather than the full `deepVogue.` path — when running scripts directly, `deepVogue/` must be on `PYTHONPATH` (handled by `pip install -e .` via `setup.py`).

### Layers (bottom → top)

1. **`deepVogue/pytorch_utils/`** — low-level infrastructure shared by every model/script.
   - `ops/` — custom CUDA ops (bias_act, upfirdn2d, conv2d_gradfix, grid_sample_gradfix, etc.) that JIT-compile.
   - `custom_ops.py`, `persistence.py` (pickling networks with their source code), `misc.py`, `training_stats.py` (multi-GPU stat reduction).

2. **`deepVogue/training/`** — model + training mechanics.
   - `networks.py` — StyleGAN2 generator/discriminator (Mapping → Synthesis, ToRGB, FromRGB, mod-demod conv).
   - `stylegan2_multi.py` — multi-resolution / multi-network variant.
   - `loss.py` — R1 + path-length regularization, ADA target.
   - `augment.py` — adaptive discriminator augmentation pipeline (`bgcfnc` etc.).
   - `dataset.py` — zip-backed image dataset + label loading.
   - `training_loop.py` — the actual loop (orchestrates G/D, EMA, snapshots, metric ticks).
   - `metrics/` — FID, KID, IS, PR, PPL.

3. **`deepVogue/neuronal_network_utils/`** and **`deepVogue/generative_network_utils/`** — fork-added helpers (`util.py`, `utilgan.py`) used by the inference/art scripts; expect lots of dnnlib-style helpers (EasyDict, formatting, file I/O).

4. **`deepVogue/dataset_tool/`** — image preprocessing into the zip format expected by `training/dataset.py`.

5. **Top-level scripts** — every "verb" is a click-decorated CLI. Mental model:
   - **Train:** `train.py` → `training_loop.py`.
   - **Sample:** `generate.py`, `style_mixing.py`, `flesh_digression.py`.
   - **Invert (image → latent):** `projector.py`, `pbaylies_projector.py`. These produce the W/W+ vectors that the *latent cinema* idea anchors on.
   - **Edit (latent → latent):** `closed_form_factorization.py` + `apply_factor.py` (SeFa-style), `blend_models.py` (layer-swap two checkpoints), `combine_npz.py`.
   - **Export / interop:** `legacy.py` (load older TF/PT pickles), `export_weights.py` (push weights to other formats), `SG2_ADA_PT_to_Rosinality.ipynb`.

6. **`notebooks/`** — exploratory & artistic. `SG2-ADA-PT_AudioReactive+Pitch.ipynb` and `Network_Blending_ADA_PT.ipynb` are the closest existing prototypes for the latent-cinema/installation direction; new artistic experiments should start as notebooks here before being promoted to scripts.

### Imports

All internal imports are package-qualified (`from deepVogue import legacy`, `from deepVogue.pytorch_utils import misc`). The original NVIDIA bare-name style (`import legacy`) was rewritten in bulk so the package works as an editable install on Colab. Mirror the package-qualified style in any new code.

The `dataset_tool.py` file at top level (NVIDIA's official `convert_dataset` CLI) and the `dataset_tool/` package both exist. The package contains `prepare.py` (the `deepvogue-prepare` CLI — inline ffmpeg frame extraction + avg-hash dedup, no external deps) and `tools.py`; the top-level file is the zip-emitter. Don't reimplement either. The dataPalette repo at `/Users/juan-garassino/Code/005-products/004-creative-tools/004-dataPalette` is *reference only* — not a runtime dependency.

## Conventions specific to this repo

- Network checkpoints are `.pkl` files containing the full module (via `pytorch_utils/persistence.py`); they embed source code, so loading old pickles requires `legacy.py`.
- Output artifacts go under `results/` (`results/checkpoints/`, `results/snapshots/`) and `deepVogue/results/` — `make clean` will wipe these, so don't keep anything precious there.
- The audio-reactive / latent-walk notebooks rely on `OpenSimplex` (already in `generate.py`'s `OSN` class) for smooth noise trajectories. Reuse `OSN` rather than rolling new noise generators when adding walk modes.
