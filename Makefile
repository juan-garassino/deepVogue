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
# R1 weight; rule of thumb ≈ 0.0002 * res^2 / batch  (256→0.5, 512→2, 1024→6.6)
DV_GAMMA       ?= 2.0
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

# Dump 20 augmented sample images for visual inspection before kicking training.
DV_PREVIEW_N ?= 20
preview-augment:
	@$(PY) -c "import random, os, sys; from pathlib import Path; \
	from PIL import Image, ImageOps; \
	from deepVogue._paths import resolve; \
	from deepVogue.dataset_tool.tarot_augment import augment_once; \
	p = resolve(); \
	out = p.walks_dir / '_aug_preview'; out.mkdir(parents=True, exist_ok=True); \
	exts = {'.png','.jpg','.jpeg','.bmp','.webp'}; \
	srcs = sorted(s for s in p.data_dir.rglob('*') if s.suffix.lower() in exts); \
	(sys.exit(f'no source images under {p.data_dir}') if not srcs else None); \
	random.seed(0); \
	[ ImageOps.fit(Image.open(random.choice(srcs)).convert('RGB'), ($(DV_RES),$(DV_RES)), Image.BICUBIC).save(out/f'orig_{i:02d}.png') for i in range(min(4,len(srcs))) ]; \
	[ augment_once(ImageOps.fit(Image.open(random.choice(srcs)).convert('RGB'), ($(DV_RES),$(DV_RES)), Image.BICUBIC)).save(out/f'aug_{i:02d}.png') for i in range($(DV_PREVIEW_N)) ]; \
	print(f'✓ wrote {$(DV_PREVIEW_N)} augmented samples to {out}')"

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
	 fail=0; \
	 for f in stylegan3-t-ffhqu-256x256.pkl stylegan3-t-metfaces-1024x1024.pkl; do \
	   if [ -f "$$dst/$$f" ] && [ $$(wc -c < "$$dst/$$f") -gt 100000000 ]; then \
	     echo "skip $$f (already present, $$(du -h "$$dst/$$f" | cut -f1))" ; \
	   else \
	     rm -f "$$dst/$$f" ; \
	     echo "fetching $$f …"; \
	     if curl -fL --progress-bar -o "$$dst/$$f" "$$base/$$f" ; then \
	       echo "✓ $$f ($$(du -h "$$dst/$$f" | cut -f1))" ; \
	     else \
	       fail=1 ; rm -f "$$dst/$$f" ; \
	       echo "✗ $$f failed — try manually:" ; \
	       echo "    wget -O $$dst/$$f $$base/$$f" ; \
	       echo "  or download from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3 and mv to $$dst/" ; \
	     fi ; \
	   fi ; \
	 done ; \
	 echo "--- $$dst ---" ; ls -lh "$$dst" ; \
	 exit $$fail

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

# Register the latest snapshot for $DV_DATASET_NAME in models.yaml.
#   make register MODEL_ID=tarot_v1
#   make register MODEL_ID=film_pretrained KIND=frames
register:
	@test -n "$(MODEL_ID)" || (echo "set MODEL_ID=<id>" && exit 1)
	@$(PY) -m deepVogue.serve.register --id "$(MODEL_ID)" \
	    --kind $${KIND:-stills} \
	    $${PKL:+--pkl $$PKL} \
	    $${ANCHORS_DIR:+--anchors-dir $$ANCHORS_DIR} \
	    $${FACTORS:+--factors $$FACTORS}

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
        download-pretrained install-serve serve register bot colab-serve film preview-augment \
        project-frames walk walk-frames walk-stills \
        factors-discover walk-factor blend eval \
        pipeline-stills pipeline-frames env count_lines

# === MLOps stack — local nano ===
.PHONY: nano-up nano-down nano-logs nano-smoke

NANO_COMPOSE := docker compose -f infra/docker-compose.yml --env-file infra/.env

nano-up: ## Bring up local MLOps stack (postgres + minio + mlflow + prefect + fastapi)
	@test -f infra/.env || (echo "create infra/.env first: cp infra/.env.example infra/.env" && exit 1)
	$(NANO_COMPOSE) up -d --build
	@echo "MLflow:   http://localhost:5000"
	@echo "Prefect:  http://localhost:4200"
	@echo "FastAPI:  http://localhost:8080"
	@echo "MinIO:    http://localhost:9001"

nano-down:
	$(NANO_COMPOSE) down -v

nano-logs:
	$(NANO_COMPOSE) logs -f --tail=200

nano-smoke: ## Run the local-nano integration smoke against a running stack
	python scripts/run_nano_smoke.py

# === MLOps stack — GCP deploy ===
.PHONY: deploy-inference deploy-mlflow deploy-prefect deploy-monitoring deploy-budget deploy-db-secrets gcp-setup gcp-setup-op publish publish-dataset show destroy

GCP_REGION ?= europe-west1
GCP_AR := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/deepvogue

publish: ## Publish latest Drive snapshot for DV_DATASET_NAME to DV_PUBLISH_TARGET as MODEL_ID
	@test -n "$(MODEL_ID)" || (echo "usage: make publish MODEL_ID=<id> [DV_DATASET_NAME=<name>]" && exit 1)
	python -m deepVogue.publish --model-id=$(MODEL_ID) \
	  --src-dir=$${DV_DRIVE_SYNC:?set DV_DRIVE_SYNC}/$${DV_DATASET_NAME:-default}

publish-dataset: ## Upload Drive dataset.zip → GCS so RunPod/Vertex can train on it (DATASET=<name>)
	@test -n "$(DATASET)" || (echo "usage: make publish-dataset DATASET=<name> DV_DATASET_DIR=<drive path> [DV_DATASET_GCS_ROOT=gs://deepvogue-datasets]" && exit 1)
	python -m deepVogue.publish dataset --name=$(DATASET) \
	  --src-dir=$${DV_DATASET_DIR:?set DV_DATASET_DIR}

deploy-inference:
	@test -n "$(GCP_PROJECT)" || (echo "set GCP_PROJECT" && exit 1)
	sed "s|PROJECT_ID|$(GCP_PROJECT)|g" infra/cloudrun/inference.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)

deploy-mlflow:
	@OP_NUMBER=$$(gcloud projects describe $${GCP_OP_PROJECT:-garassino-op} --format='value(projectNumber)' 2>/dev/null); \
	test -n "$$OP_NUMBER" || (echo "could not resolve garassino-op project number; run \`make gcp-setup-op\` first" && exit 1); \
	sed -e "s|PROJECT_ID|$(GCP_PROJECT)|g" -e "s|OP_PROJECT_NUMBER|$$OP_NUMBER|g" \
	  infra/cloudrun/mlflow.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)

deploy-prefect:
	@OP_NUMBER=$$(gcloud projects describe $${GCP_OP_PROJECT:-garassino-op} --format='value(projectNumber)' 2>/dev/null); \
	test -n "$$OP_NUMBER" || (echo "could not resolve garassino-op project number; run \`make gcp-setup-op\` first" && exit 1); \
	sed -e "s|PROJECT_ID|$(GCP_PROJECT)|g" -e "s|OP_PROJECT_NUMBER|$$OP_NUMBER|g" \
	  infra/cloudrun/prefect-server.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)
	@SERVER_URL=$$(gcloud --project=$(GCP_PROJECT) run services describe deepvogue-prefect-server \
	  --region=$(GCP_REGION) --format='value(status.url)'); \
	test -n "$$SERVER_URL" || (echo "could not resolve prefect-server URL" && exit 1); \
	sed -e "s|PROJECT_ID|$(GCP_PROJECT)|g" -e "s|PREFECT_SERVER_URL|$$SERVER_URL|g" \
	  infra/cloudrun/prefect-worker.job.yaml | \
	  gcloud --project=$(GCP_PROJECT) run jobs replace - --region=$(GCP_REGION)

gcp-setup-op: ## One-time: bootstrap garassino-op (WIF + TF state + log sink + secrets)
	@test -n "$$GITHUB_REPO" || (echo "set GITHUB_REPO=owner/repo" && exit 1)
	GCP_PROJECT=$${GCP_OP_PROJECT:-garassino-op} bash infra/gcp/setup-op.sh

gcp-setup:
	bash infra/gcp/setup.sh
	bash infra/gcp/setup-iam.sh
	bash infra/gcp/setup-db-secrets.sh || echo "db-secrets skipped (NEON_*_DSN not set; re-run \`make deploy-db-secrets\` later)"
	bash infra/gcp/setup-monitoring.sh || echo "monitoring setup failed; re-run \`make deploy-monitoring\` after deploying services"
	bash infra/gcp/setup-budget.sh || echo "budget setup skipped (billing not linked yet?)"

deploy-monitoring: ## Cloud Monitoring uptime + alerts (requires SLACK_WEBHOOK_URL)
	bash infra/gcp/setup-monitoring.sh

deploy-budget: ## €25/mo budget + 40/80/100% alerts on garassino-ml
	bash infra/gcp/setup-budget.sh

deploy-db-secrets: ## Push NEON_MLFLOW_DSN + NEON_PREFECT_DSN into op's Secret Manager
	bash infra/gcp/setup-db-secrets.sh

show: deploy-mlflow deploy-prefect deploy-inference deploy-monitoring ## Bring the stack up for a demo
	@echo
	@echo "Stack live in $(GCP_REGION):"
	@gcloud --project=$(GCP_PROJECT) run services list --region=$(GCP_REGION) \
	  --format='table(metadata.name,status.url)' 2>/dev/null || true

destroy: ## Tear down runtime only — preserves buckets, AR images, IAM, SAs. Restart with `make show`.
	@test -n "$(GCP_PROJECT)" || (echo "set GCP_PROJECT" && exit 1)
	@for svc in deepvogue-inference deepvogue-mlflow deepvogue-prefect-server; do \
	  echo "deleting service $$svc"; \
	  gcloud --project=$(GCP_PROJECT) run services delete $$svc \
	    --region=$(GCP_REGION) --quiet 2>/dev/null || true; \
	done
	@echo "deleting prefect-worker job"
	gcloud --project=$(GCP_PROJECT) run jobs delete deepvogue-prefect-worker \
	  --region=$(GCP_REGION) --quiet 2>/dev/null || true
	@echo "Runtime torn down. Buckets + AR + IAM + Neon (external) preserved."

# === RunPod training backend ===
.PHONY: runpod-build runpod-push runpod-train runpod-status runpod-logs runpod-terminate runpod-stop \
        vertex-push vertex-train vertex-status vertex-logs vertex-cancel

# NB: no '#' inside the shell call — make treats it as a comment start and
# truncates the line, leaving $(shell unterminated.
RUNPOD_IMAGE ?= ghcr.io/$(shell git config --get remote.origin.url | sed -E 's@.*[:/]([^/]+/[^/.]+)(\.git)?@\1@' | tr '[:upper:]' '[:lower:]')-train
RUNPOD_TAG ?= latest

runpod-build: ## Build the GPU train image locally
	docker build -f infra/docker/train/Dockerfile -t $(RUNPOD_IMAGE):$(RUNPOD_TAG) .

runpod-push: runpod-build ## Push the train image to GHCR
	docker push $(RUNPOD_IMAGE):$(RUNPOD_TAG)

runpod-train: ## Submit a RunPod training job (env: DV_DATASET_URI, DV_RUN_URI, GOOGLE_APPLICATION_CREDENTIALS_JSON)
	@test -n "$(MODEL_ID)" || (echo "usage: make runpod-train MODEL_ID=<id> DV_DATASET_URI=gs://... DV_RUN_URI=gs://..." && exit 1)
	DV_MODEL_ID=$(MODEL_ID) python scripts/submit_train.py --backend=runpod --model-id=$(MODEL_ID)

runpod-status: ## Print status of a pod: make runpod-status POD_ID=<id>
	@test -n "$(POD_ID)" || (echo "usage: make runpod-status POD_ID=<id>" && exit 1)
	python -c "import runpod, os, json; runpod.api_key=os.environ['RUNPOD_API_KEY']; print(json.dumps(runpod.get_pod('$(POD_ID)'), indent=2))"

runpod-logs: ## Dump current logs of a pod: make runpod-logs POD_ID=<id>
	@test -n "$(POD_ID)" || (echo "usage: make runpod-logs POD_ID=<id>" && exit 1)
	python -c "import runpod, os; runpod.api_key=os.environ['RUNPOD_API_KEY']; \
import sys; \
fn = getattr(runpod, 'get_pod_logs', None); \
sys.stdout.write(fn('$(POD_ID)') if fn else 'SDK lacks get_pod_logs; check https://www.runpod.io/console/pods')"

runpod-terminate: ## Terminate (delete) a pod: make runpod-terminate POD_ID=<id>
	@test -n "$(POD_ID)" || (echo "usage: make runpod-terminate POD_ID=<id>" && exit 1)
	python -c "import runpod, os; runpod.api_key=os.environ['RUNPOD_API_KEY']; runpod.terminate_pod('$(POD_ID)'); print('terminated $(POD_ID)')"

# ----------------------------------
#  Vertex AI training (GCP-native ephemeral GPU; VM auto-released on exit)
# ----------------------------------

VERTEX_IMAGE ?= $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/deepvogue/deepvogue-train
VERTEX_TAG ?= latest

vertex-push: runpod-build ## Tag + push the train image into Artifact Registry (Vertex must pull from AR)
	@test -n "$(GCP_PROJECT)" || (echo "usage: make vertex-push GCP_PROJECT=garassino-ml" && exit 1)
	docker tag $(RUNPOD_IMAGE):$(RUNPOD_TAG) $(VERTEX_IMAGE):$(VERTEX_TAG)
	docker push $(VERTEX_IMAGE):$(VERTEX_TAG)

vertex-train: ## Submit a Vertex AI training job (env: GCP_PROJECT, DV_DATASET_URI, DV_RUN_URI; no SA key needed)
	@test -n "$(MODEL_ID)" || (echo "usage: make vertex-train MODEL_ID=<id> GCP_PROJECT=garassino-ml DV_DATASET_URI=gs://... DV_RUN_URI=gs://..." && exit 1)
	DV_MODEL_ID=$(MODEL_ID) VERTEX_IMAGE=$(VERTEX_IMAGE):$(VERTEX_TAG) \
	    python scripts/submit_train.py --backend=vertex --model-id=$(MODEL_ID)

vertex-status: ## Describe a job: make vertex-status JOB=<custom-job resource name or numeric id>
	@test -n "$(JOB)" || (echo "usage: make vertex-status JOB=<id>" && exit 1)
	gcloud ai custom-jobs describe "$(JOB)" --project=$(GCP_PROJECT) --region=$(GCP_REGION)

vertex-logs: ## Stream logs of a job: make vertex-logs JOB=<id>
	@test -n "$(JOB)" || (echo "usage: make vertex-logs JOB=<id>" && exit 1)
	gcloud ai custom-jobs stream-logs "$(JOB)" --project=$(GCP_PROJECT) --region=$(GCP_REGION)

vertex-cancel: ## Cancel a running job: make vertex-cancel JOB=<id>
	@test -n "$(JOB)" || (echo "usage: make vertex-cancel JOB=<id>" && exit 1)
	gcloud ai custom-jobs cancel "$(JOB)" --project=$(GCP_PROJECT) --region=$(GCP_REGION)

runpod-stop: ## Pause a pod (still bills volume): make runpod-stop POD_ID=<id>
	@test -n "$(POD_ID)" || (echo "usage: make runpod-stop POD_ID=<id>" && exit 1)
	python -c "import runpod, os; runpod.api_key=os.environ['RUNPOD_API_KEY']; runpod.stop_pod('$(POD_ID)'); print('stopped $(POD_ID)')"
