# deepVogue MLOps Stack — Design

**Date:** 2026-06-03
**Author:** Juan Garassino (with Claude)
**Status:** Approved (brainstorming) → ready for implementation planning

## Goal

Wrap the existing deepVogue SG3-t pipeline (Colab training → Drive persistence → FastAPI inference → Telegram bot) with a production-shaped MLOps stack that runs locally for fast iteration and deploys piece-by-piece to GCP. Primary purpose is learning — the system must rehearse the *real* production shape, not a toy of it.

## Non-goals

- Frontend / web UI (deferred).
- Replacing Drive as Colab's training scratch (kept).
- Production-grade SLOs, autoscaling tuning, multi-region.
- Real training outside Colab in v1 (RunPod backend scaffolded but not wired).
- Prefect-driven Colab orchestration in v1 (deferred to v2).

## Constraints

- Training must continue to run in **Colab** (existing notebook), with Drive as working scratch.
- SG3-t custom CUDA ops will not compile on Mac → local nano-mode runs **mocked** train/project/walk tasks.
- All artifact code paths must accept both `s3://` (MinIO) and `gcs://` URIs unchanged (`fsspec` abstraction).
- Persistence in Drive is required for the Colab loop. GCS is the canonical store for everything downstream.

## Architecture

### Storage tiers

| Tier | Local (nano) | Prod |
|---|---|---|
| Working scratch | local FS | Drive (Colab only — `DV_DRIVE_SYNC`, unchanged) |
| Canonical artifact store | MinIO container | GCS (`gs://deepvogue-{models,datasets,walks,mlflow}`) |
| MLflow backend | Postgres container | Cloud SQL Postgres |
| MLflow artifacts | MinIO | GCS |
| Bridge | n/a | `make publish MODEL_ID=...` copies latest Drive snapshot → GCS + updates `models.yaml` |

### Compute tiers

| Job | Local nano (Mac) | Prod |
|---|---|---|
| `prepare` | real (PIL, ffmpeg) | real |
| `train` | mock (sleep + write fake-shape pkl) | Colab (manual trigger; logs to remote MLflow) |
| `publish` (Drive→GCS) | mock (MinIO copy) | real (`gsutil cp` + atomic `models.yaml` update) |
| `project` | mock | Colab |
| `walk` | mock | inference container (Cloud Run + L4 GPU) |
| `eval` | mock | Colab |

### Service map

```
LOCAL NANO (docker-compose):
  prefect-server  mlflow-server  postgres  minio  fastapi-inference
  (all on localhost network)

PROD (GCP us-central1):
  Cloud Run: prefect-server, prefect-worker, mlflow-server, fastapi-inference (L4 GPU)
  Cloud SQL: postgres (mlflow + prefect DBs)
  GCS: gs://deepvogue-{models,datasets,walks,mlflow}
  Artifact Registry: us-central1-docker.pkg.dev/<project>/deepvogue/{inference,mlflow,prefect-server,prefect-worker,train}
  Colab: training (logs to MLflow via public URL)
  GitHub Actions: build + push + deploy + Slack
```

## Services

### 1. mlflow-server
- Image base `python:3.11-slim` + `mlflow[extras]` + `psycopg2-binary` + `google-cloud-storage` + `boto3`
- Backend store via `MLFLOW_BACKEND_STORE_URI` (Postgres URL)
- Artifact store via `MLFLOW_ARTIFACTS_DESTINATION` (`s3://deepvogue-mlflow` w/ MinIO endpoint locally; `gs://deepvogue-mlflow` in prod)
- Prod auth: GOOGLE_APPLICATION_CREDENTIALS mounted as Cloud Run secret; runtime SA `mlflow-server-sa` (roles: `storage.objectAdmin` on mlflow bucket, `cloudsql.client`)
- Access: IAP-protected Cloud Run URL. Human use = Google login. Programmatic use (Colab training_loop, Prefect workers) = service-account ID token in `Authorization: Bearer <token>` header (Colab cell calls `gcloud auth print-identity-token --audiences=<iap-oauth-client-id>` and sets `MLFLOW_TRACKING_TOKEN`).
- Local port 5000 | Prod Cloud Run URL via IAP

### 2. prefect-server (self-hosted)
- Image `prefecthq/prefect:2-python3.11`
- Same Postgres instance, separate DB
- Local port 4200 | Prod Cloud Run
- Prod worker: separate Cloud Run job `prefect-worker` (process pool worker, min 0, scales on submission)
- Local nano: one container runs both server and worker

### 3. fastapi-inference (existing `deepVogue/serve/`)
- Image base **swap** `nvcr.io/nvidia/pytorch:20.12-py3` → `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (SG3 needs PyTorch 2.x)
- `pip install -e . -r requirements-serve.txt`
- Reads `models.yaml` from GCS (env `DV_MODELS_YAML=gs://deepvogue-models/models.yaml`), hot-reloads on SIGHUP or `POST /admin/reload`
- Pulls `.pkl` from GCS on first use, LRU(2) on-disk cache (existing loader)
- Prod: Cloud Run, 1×L4 GPU, 16 GB RAM, concurrency 1, timeout 600s, min instances 0
- Local nano: same image, `DV_MODELS_YAML=s3://deepvogue-models/models.yaml` (MinIO); serves a stub SG3-t state dict (img_resolution=64, fmaps=16, ~5 MB) shipped at `tests/fixtures/stub_sg3_state_dict.pt` so forward pass works without CUDA

### 4. postgres / Cloud SQL
- Two DBs: `mlflow`, `prefect`
- Prod: db-f1-micro, private IP, VPC connector for Cloud Run egress
- Local: single container, no auth, on the compose network

### 5. minio (local only)
- `minio/minio` image, port 9000 API + 9001 console
- Buckets auto-created at compose-up via init container: `deepvogue-models`, `deepvogue-datasets`, `deepvogue-walks`, `deepvogue-mlflow`
- Prod: replaced by GCS with the same bucket names

### Cross-cutting

- **`deepVogue/clients.py`** (new): `get_artifact_fs()` returns an `fsspec` filesystem honoring `DV_ARTIFACT_BACKEND=s3|gcs` plus endpoint env vars. Used everywhere artifacts are read or written.
- **`deepVogue/tracking/mlflow_helpers.py`** (new): `log_training_run(run_dir)` reads existing `metric-fid50k_full.jsonl` + snapshot pkls and logs to MLflow. Called from Colab training_loop at every tick and at end.

## Prefect flows + data flow

All flows in `deepVogue/orchestration/flows.py`. Each task takes `backend: Literal["local","colab","runpod"]` and dispatches to `deepVogue/orchestration/backends/{local,colab,runpod}.py`.

### Flow inventory

| Flow | Inputs | Outputs | Backends |
|---|---|---|---|
| `prepare_flow` | `source_uri`, `dataset_name`, `res`, `kind=stills\|frames`, `fps?` | `dataset.zip` + `frames_index.json` to GCS/MinIO | local (real) |
| `train_flow` | `dataset_name`, `cfg=stylegan3-t`, `kimg`, `gamma`, `batch`, `resume_from?` | `network-snapshot-*.pkl` series to GCS | local (mock); colab + runpod stubs in v1 |
| `publish_flow` | `model_id`, `src_path` (Drive or GCS) | row appended to `models.yaml` on GCS, Slack notify | local (real) |
| `project_flow` | `model_id`, `frames_uri`, `stride`, `steps` | per-frame `projected_w.npz` to GCS | local (mock); colab stub in v1 |
| `walk_flow` | `model_id`, `anchors_uri?`, `seeds?`, `mode=cubic`, `steps`, `fps`, `factor_idx?`, `factor_amp?` | `walk_id.mp4` to GCS | local (mock); inference-server HTTP (real, prod) |
| `eval_flow` | `model_id`, `dataset_uri` | FID + KID logged to MLflow | local (mock); colab stub in v1 |
| `pipeline_stills` | top-level | composes prepare → train → publish → walk | dispatches per-task |
| `pipeline_frames` | top-level | composes prepare → train → publish → project → walk → eval | dispatches per-task |

### Backend behaviors

**`backends/local.py`** (nano-mode):
- `prepare`: real. PIL/ffmpeg → MinIO upload.
- `train`: stub. `time.sleep(kimg / 50)`, writes a zero-tensor pkl with the SG3-t-64 stub state dict shape. Logs fake-but-monotonically-decreasing FID to MLflow.
- `project`, `walk`, `eval`: stubs writing valid-shape `.npz`, 1-frame `.mp4`, FID number.

**`backends/colab.py`** (v2, stubbed in v1):
- v2 plan: drops `training_request.json` to `gs://deepvogue-queue/`, returns Prefect `Suspended` state. Colab polling cell picks up requests, runs `make train`, writes `training_complete.json` with snapshot path. Separate Prefect deployment `watch_colab_completions` polls and resumes suspended runs.
- v1: raises `NotImplementedError("use manual Colab + `make publish` in v1")`.

**`backends/runpod.py`** (v2, stubbed):
- v2 plan: RunPod GraphQL API spins up an L4/A100 pod with the training image, `gsutil cp` results, terminate.
- v1: raises `NotImplementedError`.

### Data flow — full pipeline (stills, prod)

```
[user] make publish MODEL_ID=tarot_v1
   │
   ▼
Colab Drive snapshot ──(gsutil)──► gs://deepvogue-models/tarot_v1/snap.pkl
                                       │
                                       ▼ appends row (atomic via temp + rename)
                              gs://deepvogue-models/models.yaml
                                       │
                                       ▼ SIGHUP / POST /admin/reload
                              fastapi-inference (Cloud Run)
                                       ▲
                                       │ /walk POST
[Prefect walk_flow] ───────────────────┘
        │                              │
        │ on success                   │ .mp4 written
        ▼                              ▼
    MLflow run logged             gs://deepvogue-walks/tarot_v1/<walk_id>.mp4
        │                              │
        ▼                              ▼
    Slack #deepvogue-runs ─── "✓ walk tarot_v1/abc done — <signed url>"
```

### MLflow taxonomy
- Experiment per dataset: `tarot`, `film`, `tarot_pretrained`, …
- Run per training session (Colab → remote MLflow): params (`cfg`, `gamma`, `kimg`, `batch`, `res`), metrics (`fid50k_full` per snapshot), artifacts (every Nth snapshot pkl, sample image grid).
- Run per Prefect flow: linked via `mlflow.set_tag("prefect_flow_run_id", ...)` so every generated walk traces back to the model run it used.

### Drive ↔ GCS bridge
- `make publish MODEL_ID=<id>` (wraps `deepVogue/publish.py`): walks latest `network-snapshot-*.pkl` under `$DV_DRIVE_SYNC/$DV_DATASET_NAME/`, validates with `legacy.load_network_pkl`, uploads to GCS, atomically updates `models.yaml` (download → mutate → upload with `if-generation-match` to prevent races), posts Slack.
- Run from Colab post-training or from Mac after `rclone sync drive: ./tmp`.

### models.yaml schema change
Existing schema (per CLAUDE.md) uses **relative** `pkl:` paths resolved against `$DV_RUN_DIR`. New schema accepts **absolute** URIs (`gs://...`, `s3://...`) so the same registry works for prod (GCS) and nano (MinIO). Existing relative paths remain supported (resolved against `DV_MODELS_ROOT` env var, defaulting to current behavior) — backwards compatible.

## CI/CD + Slack

### GitHub Actions workflows (`.github/workflows/`)

| Workflow | Trigger | Behavior | Slack |
|---|---|---|---|
| `test.yml` | push, PR to master | `pytest` + `black --check` + import smoke (no GPU) | post on PR fail only |
| `build-inference.yml` | push to master touching `deepVogue/serve/**`, `infra/docker/inference/**`, or manual | build → push AR → deploy Cloud Run → `/health` smoke → traffic 100% if green | start / success / fail |
| `build-mlflow.yml` | same pattern for `infra/docker/mlflow/**` | same shape | start / success / fail |
| `build-prefect.yml` | same for `infra/docker/prefect/**` | builds server + worker images, deploys both | start / success / fail |
| `build-train.yml` | manual only | builds training image (RunPod v2) → push AR | success / fail |
| `nano-smoke.yml` | nightly cron + PR | `docker compose -f infra/docker-compose.yml up -d` → run `pipeline_stills_smoke` flow → assert all tasks SUCCEEDED → tear down | post on fail |

### Auth setup
- GCP: Workload Identity Federation between GitHub OIDC and a `deepvogue-deployer` SA (no JSON keys checked in). SA roles: `artifactregistry.writer`, `run.admin`, `iam.serviceAccountUser`, `cloudsql.client`.
- GitHub secrets: `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`, `GCP_PROJECT`, `SLACK_WEBHOOK_URL`.
- Runtime SAs (least-privilege): `mlflow-server-sa`, `prefect-server-sa`, `prefect-worker-sa`, `fastapi-inference-sa`.
- Local nano: no GCP creds; MinIO root user/pass in `.env`.

### Slack notifications

Channels: `#deepvogue-ci` (CI/CD) and `#deepvogue-runs` (runtime / flow events).

| Event | Channel | Payload |
|---|---|---|
| GH Actions: build start | `#ci` | branch, commit, workflow |
| GH Actions: build pass | `#ci` | duration, deployed URL, revision id |
| GH Actions: build fail | `#ci` | step that failed, log link |
| Nano-smoke nightly fail | `#ci` | flow run id, failed task, MLflow link |
| Prefect flow success (`walk`, `pipeline_*`) | `#runs` | model_id, params, signed GCS URL |
| Prefect flow failure | `#runs` | flow name, task, exception, MLflow URL |
| MLflow new run logged (Colab) | `#runs` | dataset, kimg, latest FID |
| `make publish` ran | `#runs` | model_id, snapshot path, FID |
| Cloud Run inference error spike (Cloud Monitoring alert) | `#runs` | service, error count |

Implementation: `deepVogue/notifications/slack.py` thin wrapper around `requests.post(SLACK_WEBHOOK_URL, json=blocks)`. Prefect tasks call helpers on terminal states via a Prefect hook. GitHub Actions use `slackapi/slack-github-action@v1`.

### Artifact Registry layout

```
us-central1-docker.pkg.dev/<project>/deepvogue/
  inference:<sha>, inference:latest
  mlflow:<sha>, mlflow:latest
  prefect-server:<sha>, prefect-server:latest
  prefect-worker:<sha>, prefect-worker:latest
  train:<sha>, train:latest
```

### Deploy targets
- All Cloud Run services: region `us-central1`, min instances 0, HTTPS only.
- inference: 1 L4 GPU, 16 GB RAM, concurrency 1, timeout 600s.
- mlflow/prefect-server: CPU-only, 1 vCPU / 2 GB, concurrency 80.

## Testing

| Layer | Local | CI | Manual |
|---|---|---|---|
| Unit tests (existing 22) | `pytest tests/` | `test.yml` on every push | — |
| New unit tests | + `tests/test_orchestration_backends.py` (dispatch logic), `tests/test_publish.py` (path resolution w/ tmp dirs), `tests/test_slack.py` (block formatting, mocked POST), `tests/test_clients.py` (fsspec backend resolution) | same | — |
| docker-compose smoke | `make nano-up && make nano-smoke` runs `pipeline_stills_smoke` end-to-end against MinIO+local MLflow; asserts mp4 written, MLflow run logged, ≥1 Slack call attempted (mocked webhook) | `nano-smoke.yml` nightly + PR | — |
| Inference container smoke | `docker build -f infra/docker/inference/Dockerfile .` + run with stub model | `build-inference.yml` post-deploy `/health` curl | — |
| Real training | — | — | Colab notebook cell |
| Real inference | — | — | `curl <prod-url>/walk` after first publish |

**Mock model fixture** (`tests/fixtures/stub_sg3_state_dict.pt`): tiny but structurally valid SG3-t state dict (z_dim=512, w_dim=512, img_resolution=64, img_channels=3, fmaps capped at 16), ~5 MB, generated once by `scripts/build_stub_state_dict.py` and checked in.

## Decisions (resolved)

1. **Inference GPU**: Cloud Run w/ L4 (scales-to-zero, ~$0.71/hr active).
2. **Prefect topology**: self-hosted (Cloud Run server + worker job).
3. **MLflow access**: IAP-protected Cloud Run URL.
4. **Drive→GCS publish trigger**: manual `make publish` only in v1.
5. **Colab manual-handoff queue**: v2 (Colab + RunPod backends are scaffolded but stubbed in v1).
6. **v1 scope**: local docker-compose nano, MLflow-from-Colab, GCP deploy of inference + MLflow + Prefect, CI/CD + Slack. Prefect training backend mocked-only.

## Repo additions

```
infra/
  docker-compose.yml
  .env.example
  docker/
    inference/Dockerfile
    mlflow/Dockerfile
    prefect/Dockerfile          # multi-target: server, worker
    train/Dockerfile            # v2 RunPod-ready
  cloudrun/
    inference.service.yaml
    mlflow.service.yaml
    prefect-server.service.yaml
    prefect-worker.job.yaml
  gcp/
    setup.sh                    # enable APIs, create buckets, AR repo, Cloud SQL, VPC connector (for Cloud Run → Cloud SQL private IP), runtime SAs, WIF

deepVogue/
  clients.py                    # fsspec helpers
  publish.py                    # Drive→GCS + models.yaml update
  tracking/
    __init__.py
    mlflow_helpers.py           # log_training_run, log_flow_run
  orchestration/
    __init__.py
    flows.py                    # prepare/train/publish/project/walk/eval + pipelines
    backends/
      __init__.py
      local.py                  # nano mocks
      colab.py                  # v2 stub
      runpod.py                 # v2 stub
  notifications/
    __init__.py
    slack.py

.github/workflows/
  test.yml
  build-inference.yml
  build-mlflow.yml
  build-prefect.yml
  build-train.yml               # manual
  nano-smoke.yml

requirements-orchestration.txt  # prefect, mlflow, fsspec[gcs,s3], slack-sdk
tests/
  test_orchestration_backends.py
  test_publish.py
  test_slack.py
  test_clients.py
  fixtures/stub_sg3_state_dict.pt
scripts/
  build_stub_state_dict.py
docs/superpowers/specs/2026-06-03-mlops-deepvogue-design.md
```

### New Makefile targets

- `make nano-up` — `docker compose -f infra/docker-compose.yml up -d`
- `make nano-down` — tear down
- `make nano-smoke` — run `pipeline_stills_smoke` Prefect deployment locally
- `make publish MODEL_ID=...` — Drive → GCS + `models.yaml` update + Slack
- `make deploy-inference` / `deploy-mlflow` / `deploy-prefect` — local-triggered Cloud Run deploy (CI/CD also does this)
- `make gcp-setup` — runs `infra/gcp/setup.sh`

## Docs to update (per docs-current protocol)

- **CLAUDE.md** — add section "MLOps stack" after "Serving & bot". Document `infra/`, env vars (`MLFLOW_TRACKING_URI`, `PREFECT_API_URL`, `SLACK_WEBHOOK_URL`, `DV_ARTIFACT_BACKEND`, `MINIO_*`, `GCP_PROJECT`), GCS bucket layout, new Makefile targets (`nano-up`, `nano-smoke`, `publish`, `deploy-*`, `gcp-setup`).
- **README.md** — add "Local nano-mode" quickstart (`cp infra/.env.example .env && make nano-up && make nano-smoke`) and "GCP deployment" pointer to `infra/gcp/setup.sh`.
- **DOCS.md** (root catalog at `../../DOCS.md`) — update deepVogue entry to mention MLflow/Prefect/Cloud Run inference deployment.
- No new docs created — all updates surgical to existing files.

## Open / deferred (v2 backlog)

- Colab ↔ Prefect handoff queue (`backends/colab.py`)
- RunPod backend (`backends/runpod.py`) + `infra/docker/train/Dockerfile`
- Frontend web UI
- Auto-publish on Drive snapshot rotation
- Cloud Monitoring alerts → Slack
- Cost dashboard (BigQuery export of billing → Looker Studio)
