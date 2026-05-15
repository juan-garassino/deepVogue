# ----------------------------------
#  deepVogue — Colab/Drive runtime
# ----------------------------------
# Pattern shared with /Users/juan-garassino/Code/005-products/010-more-than-words.
# Datasets and outputs live on Drive in Colab; the same DV_* env vars drive
# local dry-runs against /tmp paths.

ENV := PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY  := $(ENV) PYTHONPATH=. python3

# Path config (override on the command line or via env).
# DV_DATA_DIR / DV_DATASET_DIR / DV_RUN_DIR / DV_DRIVE_SYNC / DV_NETWORK_PKL
# DV_ANCHORS_DIR / DV_WALKS_DIR
DV_RES         ?= 512
DV_FPS         ?= 1
DV_KIMG        ?= 5000
DV_GPUS        ?= 1
DV_PROJ_STEPS  ?= 500
DV_PROJ_STRIDE ?= 4
DV_INTERP      ?= slerp
DV_NOISE       ?= osn
DV_WALK_FPS    ?= 24
DV_FPS_PER_SEG ?= 24
DV_FACTOR_IDX  ?= 7
DV_FACTOR_AMP  ?= 0.4
DV_BLEND_RES   ?= 32
DV_CFG         ?= stylegan3-t
DV_BATCH       ?= 32
DV_GAMMA       ?= 8.2
DV_AUG         ?= ada
DV_MIRROR      ?= 1
DV_AUGMENT     ?= 0            # set to 1 for procedural augmentation on stills
DV_MAX_AUG     ?= 10000

# Default GitHub remote (override via env or `make colab-clone GITHUB_URL=...`)
DV_GITHUB_URL  ?= https://github.com/JuanGarassino/deepVogue.git

# ----------------------------------
#  install / lint / test
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

install_train_requirements:
	@pip install -r requirements.txt -r requirements-train.txt

# Run from inside Colab after cloning the repo.
colab-install:
	@pip install -q -e . -r requirements-train.txt
	@apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true

# Clone (or fast-forward) the repo into /content/deepVogue. Run this from a fresh
# Colab cell *before* cd'ing in: `!make -f /tmp/Makefile colab-clone` is the
# bootstrap pattern; see notebooks/deepVogue_colab.ipynb cell 00.
colab-clone:
	@if [ -d /content/deepVogue/.git ]; then \
	   cd /content/deepVogue && git pull --ff-only ; \
	 else \
	   git clone --depth 1 $(DV_GITHUB_URL) /content/deepVogue ; \
	 fi

check_code:
	@flake8 deepVogue/

black:
	@black deepVogue/ tests/

test:
	@coverage run -m pytest tests/
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

clean:
	@rm -f .coverage
	@rm -fr build dist deepVogue.egg-info deepVogue-*.dist-info
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ----------------------------------
#  data prep (inline ffmpeg + PIL)
# ----------------------------------

prepare-stills:
	$(PY) -m deepVogue.dataset_tool.prepare stills \
	    --resolution $(DV_RES) \
	    $$( [ "$(DV_AUGMENT)" = "1" ] && echo "--augment-procedural --max-augmented $(DV_MAX_AUG)" )

prepare-frames:
	$(PY) -m deepVogue.dataset_tool.prepare frames \
	    --resolution $(DV_RES) --fps $(DV_FPS)

# ----------------------------------
#  training
# ----------------------------------

train:
	$(PY) -m deepVogue.train \
	    --outdir $$DV_RUN_DIR \
	    --cfg $(DV_CFG) \
	    --data $$DV_DATASET_DIR/dataset.zip \
	    --gpus $(DV_GPUS) --batch $(DV_BATCH) --gamma $(DV_GAMMA) \
	    --mirror $(DV_MIRROR) --aug $(DV_AUG) --kimg $(DV_KIMG) \
	    $${DV_NETWORK_PKL:+--resume $$DV_NETWORK_PKL}

# Pull NVIDIA SG3-t pretrained checkpoints into Drive for fine-tune resume.
# Set DV_PRETRAINED_DIR (or fall back to $DV_DRIVE_SYNC/../pretrained).
download-pretrained:
	@dst=$${DV_PRETRAINED_DIR:-$$DV_DRIVE_SYNC/../pretrained}; mkdir -p "$$dst"; \
	 base=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files; \
	 for f in stylegan3-t-ffhqu-256x256.pkl stylegan3-t-metfaces-1024x1024.pkl; do \
	   if [ -f "$$dst/$$f" ]; then echo "skip $$f"; else \
	   echo "fetching $$f"; \
	   curl -L -o "$$dst/$$f" "$$base/$$f" || echo "(failed $$f — pull manually from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3)"; fi; \
	 done; \
	 ls -lh "$$dst"

sync-out:
	$(PY) -c "from deepVogue._drive_sync import DriveSync; from deepVogue._paths import resolve; p=resolve(); DriveSync(p.run_dir, p.drive_sync, 1)._mirror_once()"

latest-pkl:
	@$(PY) -c "from deepVogue._paths import latest_snapshot; pkl=latest_snapshot(); print(pkl or '')"

# Resume training from latest Drive snapshot for $DV_DATASET_NAME.
resume:
	@PKL=$$($(PY) -c "from deepVogue._paths import latest_snapshot; p=latest_snapshot(); print(p or '')"); \
	 if [ -z "$$PKL" ]; then echo "no snapshot found for DV_DATASET_NAME=$$DV_DATASET_NAME"; exit 1; fi; \
	 echo "resuming from $$PKL"; \
	 DV_NETWORK_PKL="$$PKL" $(MAKE) train

# Print models.yaml (registry) from Drive.
models-list:
	@$(PY) -c "from deepVogue._paths import resolve; from pathlib import Path; import os; p=resolve(); base=p.drive_sync.parent if p.drive_sync else p.run_dir.parent; y=Path(os.environ.get('DV_MODELS_YAML') or (base/'models.yaml')); print(y.read_text() if y.exists() else f'(no registry at {y})')"

# ----------------------------------
#  latent cinema — projection + walk
# ----------------------------------

# Project every DV_PROJ_STRIDE-th frame from frames_index.json into W+ space.
project-frames:
	$(PY) -m deepVogue.cinema project-frames --stride $(DV_PROJ_STRIDE) --num-steps $(DV_PROJ_STEPS)

# Render walk.mp4 from anchors using either frames_index.json (movies) or alphabetical order (stills).
walk:
	$(PY) -m deepVogue.walk \
	    --network $$DV_NETWORK_PKL \
	    --anchors $$DV_ANCHORS_DIR \
	    --out     $$DV_WALKS_DIR/walk.mp4 \
	    --interp  $(DV_INTERP) --noise-mode $(DV_NOISE) \
	    --fps $(DV_WALK_FPS) --frames-per-segment $(DV_FPS_PER_SEG) \
	    $${DV_DATASET_DIR:+--order $$DV_DATASET_DIR/frames_index.json}

walk-frames: walk
walk-stills:
	$(PY) -m deepVogue.walk \
	    --network $$DV_NETWORK_PKL \
	    --anchors $$DV_ANCHORS_DIR \
	    --out     $$DV_WALKS_DIR/walk.mp4 \
	    --interp  $(DV_INTERP) --noise-mode $(DV_NOISE) \
	    --fps $(DV_WALK_FPS) --frames-per-segment $(DV_FPS_PER_SEG)

# ----------------------------------
#  SeFa factors + blend (Refik-style edits)
# ----------------------------------

factors-discover:
	$(PY) -m deepVogue.factors discover \
	    $${DV_NETWORK_PKL:+--network $$DV_NETWORK_PKL} \
	    --out $$DV_RUN_DIR/factors.pt

# Re-render the walk with a SeFa direction ramp.
walk-factor:
	$(PY) -m deepVogue.walk \
	    --network $$DV_NETWORK_PKL \
	    --anchors $$DV_ANCHORS_DIR \
	    --out     $$DV_WALKS_DIR/walk_factor$(DV_FACTOR_IDX).mp4 \
	    --interp  $(DV_INTERP) --noise-mode $(DV_NOISE) --fps $(DV_WALK_FPS) \
	    --factor-drift $$DV_RUN_DIR/factors.pt \
	    --factor-index $(DV_FACTOR_IDX) --factor-amp $(DV_FACTOR_AMP) \
	    $${DV_DATASET_DIR:+--order $$DV_DATASET_DIR/frames_index.json}

blend:
	@test -n "$(LOW)"  || (echo "set LOW=<low-res-pkl>"  && exit 1)
	@test -n "$(HIGH)" || (echo "set HIGH=<hi-res-pkl>" && exit 1)
	$(PY) -m deepVogue.blend \
	    --low-res-pkl $(LOW) --hi-res-pkl $(HIGH) \
	    --resolution $(DV_BLEND_RES) \
	    --out $$DV_RUN_DIR/blended.pkl

# ----------------------------------
#  evaluation
# ----------------------------------

# Print the FID curve from $DV_RUN_DIR/metric-fid50k_full.jsonl.
eval:
	$(PY) -m deepVogue.cinema eval-fid

# ----------------------------------
#  end-to-end pipelines (one command per dataset type)
# ----------------------------------

pipeline-stills: prepare-stills train walk-stills

pipeline-frames: prepare-frames train project-frames walk eval

# Render a film as a latent walk: pick latest pkl, project anchors, write mp4 on Drive.
# Usage: make film DV_DATASET_NAME=film FILM_ID=test30s
film:
	@test -n "$(FILM_ID)" || (echo "set FILM_ID=<name>" && exit 1)
	@PKL=$$($(PY) -c "from deepVogue._paths import latest_snapshot; p=latest_snapshot(); print(p or '')"); \
	 test -n "$$PKL" || (echo "no snapshot for DV_DATASET_NAME=$$DV_DATASET_NAME" && exit 1); \
	 echo "[film] checkpoint = $$PKL"; \
	 DV_NETWORK_PKL="$$PKL" $(PY) -m deepVogue.cinema project-frames --stride $(DV_PROJ_STRIDE) --num-steps $(DV_PROJ_STEPS); \
	 mkdir -p $$DV_WALKS_DIR; \
	 DV_NETWORK_PKL="$$PKL" $(PY) -m deepVogue.walk \
	   --network "$$PKL" --anchors $$DV_ANCHORS_DIR \
	   --order $$DV_DATASET_DIR/frames_index.json \
	   --out $$DV_WALKS_DIR/$(FILM_ID).mp4 \
	   --interp cubic --fps $(DV_WALK_FPS) --frames-per-segment $(DV_PROJ_STRIDE) ; \
	 echo "✓ $$DV_WALKS_DIR/$(FILM_ID).mp4"

# ----------------------------------
#  serve (FastAPI + Telegram bot)
# ----------------------------------

DV_FASTAPI_HOST ?= 0.0.0.0
DV_FASTAPI_PORT ?= 8080

install-serve:
	@pip install -q -r requirements-serve.txt -r requirements-bot.txt

serve:
	@$(PY) -m uvicorn deepVogue.serve.api:app --host $(DV_FASTAPI_HOST) --port $(DV_FASTAPI_PORT)

# Run the Telegram bot (long-polling). Needs DV_TG_TOKEN, optional DV_TG_ALLOWLIST.
bot:
	@$(PY) -m deepVogue.bot.telegram

# Run FastAPI + Telegram bot together in one Colab cell. Opens an ngrok tunnel
# to FastAPI in case you want to hit it from anywhere; the bot itself reaches
# the API over localhost.
colab-serve:
	@$(PY) -c "from pyngrok import ngrok; print('ngrok ->', ngrok.connect($(DV_FASTAPI_PORT)).public_url)" || echo "(install pyngrok if you want the tunnel)"
	@($(PY) -m uvicorn deepVogue.serve.api:app --host $(DV_FASTAPI_HOST) --port $(DV_FASTAPI_PORT) &) ; \
	 sleep 3 ; \
	 $(PY) -m deepVogue.bot.telegram

# ----------------------------------
#  introspection
# ----------------------------------

env:
	$(PY) -c "from deepVogue._paths import env_summary; print(env_summary())"

count_lines:
	@find ./deepVogue -name '*.py' -exec wc -l {} \; | sort -n | awk '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'

.PHONY: install_requirements install_train_requirements colab-install colab-clone \
        check_code black test clean \
        prepare-stills prepare-frames train resume sync-out latest-pkl models-list \
        download-pretrained install-serve serve bot colab-serve film \
        project-frames walk walk-frames walk-stills \
        factors-discover walk-factor blend eval \
        pipeline-stills pipeline-frames env count_lines
