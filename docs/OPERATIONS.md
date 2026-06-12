# deepVogue MLOps — Operations

Single-page runbook. Quickstart → daily ops → training paths → failure modes → architecture decisions → cost. Designed to be re-readable after a 2-week absence.

The stack lives in **`garassino-ml`** (deepVogue runtime + data) and federates auth from **`garassino-op`** (control plane: WIF pool, TF state, Secret Manager, log sink). Region `europe-west1` everywhere. Cost cap €25/mo with 40/80/100% alerts.

---

## 1. Cold-start sequence (once per workspace)

```bash
# Prereqs: gcloud authed, both projects created with billing linked,
# GHCR token in `gh auth login`, repo cloned at this worktree.

# 1. Bootstrap garassino-op — WIF, TF state bucket, log sink, secret placeholders
GCP_PROJECT=garassino-op GITHUB_REPO=<owner>/deepvogue make gcp-setup-op

# 2. Mint a Neon project (neon.tech free tier) → grab two DSNs

# 3. Bootstrap garassino-ml — runtime SAs, AR repo, monitoring, budget,
#    cross-project WIF binding to op. Then push the Neon DSNs.
export GCP_PROJECT=garassino-ml GITHUB_REPO=<owner>/deepvogue
make gcp-setup
NEON_MLFLOW_DSN='postgresql://...' NEON_PREFECT_DSN='postgresql://...' \
  make deploy-db-secrets

# 4. Register GitHub repo secrets (Settings → Secrets → Actions):
#    GCP_PROJECT=garassino-ml
#    GCP_WIF_PROVIDER=<value printed by setup-op.sh>
#    GCP_DEPLOYER_SA=deepvogue-deployer-sa@garassino-ml.iam.gserviceaccount.com
#    SLACK_WEBHOOK_URL=<webhook>
#    TELEGRAM_BOT_TOKEN=<DV_TG_TOKEN>
#    TELEGRAM_CHAT_ID=<from getUpdates>

# 5. Wire the budget email recipient once in the console:
#    Billing → Budgets → deepvogue-25-eur → Manage notifications →
#    add juan.garassino@hotmail.com

# 6. Mint the RunPod trainer SA key (documented WIF exception — external compute)
gcloud iam service-accounts keys create trainer-key.json \
  --iam-account=deepvogue-trainer-sa@garassino-ml.iam.gserviceaccount.com
gcloud --project=garassino-op secrets versions add op-trainer-key \
  --data-file=trainer-key.json && rm trainer-key.json
```

After this, the workspace is provisioned and CI can deploy.

---

## 2. Daily ops

| I want to… | Command |
|---|---|
| Bring up the demo stack | `make show` (~1 min — deploys mlflow + prefect + inference + monitoring) |
| Tear it back down | `make destroy` (services + jobs gone; buckets + AR + Neon preserved) |
| Verify everything is up | `make show` final block prints Cloud Run URLs |
| Re-deploy one service | `make deploy-{mlflow,prefect,inference}` |
| Refresh monitoring | `make deploy-monitoring` |
| Refresh budget | `make deploy-budget` |
| Push new Neon DSNs | `NEON_MLFLOW_DSN=… NEON_PREFECT_DSN=… make deploy-db-secrets` |
| List published models | `make models-list` |
| Tail a Cloud Run service | `gcloud run services logs read deepvogue-<svc> --region=europe-west1` |

---

## 3. Training paths

**Three paths, same artifacts.** All land snapshots in `gs://deepvogue-runs/<dataset>/` and publish to `gs://deepvogue-models/models.yaml`.

**Data flow:** datasets are prepared on Colab against Drive (`make prepare-stills` / `prepare-frames`). RunPod and Vertex jobs can't read Drive — bridge once per dataset with `make publish-dataset DATASET=<name>` (run it in a Colab cell after prepare; uploads `dataset.zip` + `frames_index.json` from `$DV_DATASET_DIR` to `gs://deepvogue-datasets/<name>.zip` and prints the `DV_DATASET_URI` to pass to the train job). Datasets are immutable zips, so this is a one-time copy, not a sync.

### Colab (interactive; v1 default)

Open `notebooks/deepVogue_colab.ipynb` on Colab. Cells 00–07 form a linear pipeline: setup → configure → prepare → train → project anchors → walk → factors → eval. Drive holds data + outputs; the code is git-cloned each session.

- Mid-training disconnects survive: `make resume` looks at the latest snapshot under `$DV_DRIVE_SYNC/<DV_DATASET_NAME>/` and restarts with `DV_NETWORK_PKL` set
- After train completes, the notebook's `make publish MODEL_ID=<id>` bridges Drive → GCS + appends to `models.yaml`
- Gamma rule of thumb: `≈ 0.0002 * res² / batch` (e.g. 512px @ batch=32 → gamma=2)

### RunPod (autonomous; v1 backend)

```bash
# Prereq once: image pushed to GHCR via CI (auto-fires on push to mlops/v1)
export RUNPOD_API_KEY=<from runpod.io/console/user/settings>
export RUNPOD_IMAGE=ghcr.io/<owner>/deepvogue-train:latest
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat trainer-key.json)"

# Submit
make runpod-train \
  MODEL_ID=tarot DV_KIMG=5000 DV_GAMMA=2 DV_BATCH=32 DV_RES=512 \
  DV_DATASET_URI=gs://deepvogue-datasets/tarot.zip \
  DV_RUN_URI=gs://deepvogue-runs/tarot \
  DV_PUBLISH_TARGET=gs://deepvogue-models

# Returns a pod_id immediately on stdout (JSON). Then blocks until done or
# RUNPOD_MAX_TRAIN_HOURS (default 24) elapses.

# Ops while running
make runpod-status POD_ID=<id>
make runpod-logs   POD_ID=<id>
make runpod-terminate POD_ID=<id>   # idempotent
```

The pod self-terminates via RunPod GraphQL `podTerminate` on entrypoint exit (success or failure). Snapshots mirror to GCS every 60s while training runs; final pkl + FID land in `models.yaml` via `publish_checkpoint`.

### Vertex AI (GCP-native; same train image, no SA key)

The same train container runs as a Vertex AI CustomJob: GCP provisions the GPU
VM, runs the container to completion, and **releases the VM automatically** —
zero idle cost and nothing to reap. Auth is ambient ADC via the attached
trainer SA (the entrypoint detects the missing
`GOOGLE_APPLICATION_CREDENTIALS_JSON` and uses the metadata server), so the
RunPod SA-key exception does not apply here.

```bash
# Prereq once: `make gcp-setup` (enables aiplatform API + IAM bindings),
# then push the train image into Artifact Registry (Vertex can't pull GHCR):
make vertex-push GCP_PROJECT=garassino-ml

# Submit (launcher needs gcloud auth with roles/aiplatform.user)
make vertex-train \
  MODEL_ID=tarot GCP_PROJECT=garassino-ml DV_KIMG=5000 DV_GAMMA=2 DV_BATCH=32 DV_RES=512 \
  DV_DATASET_URI=gs://deepvogue-datasets/tarot.zip \
  DV_RUN_URI=gs://deepvogue-runs/tarot \
  DV_PUBLISH_TARGET=gs://deepvogue-models

# Returns the customJobs resource name immediately (JSON line), then blocks
# until the job ends or VERTEX_MAX_TRAIN_HOURS (default 24) elapses.

# Ops while running
make vertex-status JOB=<id>
make vertex-logs   JOB=<id>
make vertex-cancel JOB=<id>
```

GPU tiers (europe-west1): default `NVIDIA_L4` on `g2-standard-8` (~€0.85/hr,
24 GB); cheap tier `VERTEX_ACCELERATOR=NVIDIA_TESLA_T4 VERTEX_MACHINE_TYPE=n1-standard-8`
(~€0.55/hr, 16 GB). A100s live in europe-west4 (`GCP_REGION=europe-west4`).
GPU quota for Vertex (`custom_model_training_nvidia_l4_gpus`) must be granted
once per project — request it in IAM → Quotas if the first submit fails with
a quota error.

Pick RunPod for big GPUs at spot prices (H100); pick Vertex when you want the
job inside `garassino-ml` (budget alerts cover it, logs in Cloud Logging, no
SA key to mint or rotate).

---

## 4. Failure modes (runbook)

| Symptom | Likely cause | Fix |
|---|---|---|
| `make gcp-setup-op` fails: project not found | `garassino-op` not created | Create the project in console + link billing first |
| `make gcp-setup` errors: "cannot resolve garassino-op" | Step 1 not run | Run `make gcp-setup-op` |
| `make deploy-db-secrets` skips silently | `NEON_*_DSN` env vars unset | Mint Neon project, export both DSNs, re-run |
| `make show` Cloud Run fails: image not found in AR | CI hasn't pushed yet | Push `mlops/v1` to GitHub; `build-{mlflow,prefect,inference}.yml` will run |
| MLflow service crashes: secret access denied | `mlflow-server-sa` missing `secretAccessor` on `op-neon-mlflow-dsn` | Re-run `make deploy-db-secrets` |
| RunPod pod stuck "RUNNING" indefinitely | Container failed to pull GHCR image (private + no auth) | Confirm GHCR package visibility is **public** (CI step does this; first run may need manual flip in GitHub Packages settings) |
| RunPod pod errors at boot: `gsutil 401` | Bad SA JSON or expired key | Re-mint trainer-sa key, re-upsert into `op-trainer-key` |
| `make vertex-train` fails: quota exceeded | No Vertex GPU quota in region | IAM → Quotas → request `custom_model_training_nvidia_l4_gpus` (or T4 equivalent) for europe-west1 |
| Vertex job fails at boot: image pull error | Train image not in Artifact Registry | `make vertex-push GCP_PROJECT=garassino-ml` (Vertex can't pull GHCR) |
| `make vertex-train` Python error: no module `google.cloud.aiplatform` | SDK not installed in caller env | `pip install google-cloud-aiplatform>=1.60.0` |
| RunPod pod errors at boot: `assert torch.cuda.is_available()` | GPU type unavailable in region; pod got CPU | Re-submit with a more common GPU SKU (`RUNPOD_GPU_TYPE="NVIDIA A40"`) |
| RunPod pod errors at boot: custom ops compile fail | nvcc/torch mismatch | Switch train Dockerfile base to `nvidia/cuda:12.1.0-devel-ubuntu22.04` + manual torch install |
| `make runpod-train` Python error: no module `runpod` | SDK not installed in caller env | `pip install runpod>=1.6.0` |
| Inference service crashes on first model load | nvcc missing or torch wrong | Should not happen on devel base; check `infra/docker/inference/Dockerfile` wasn't accidentally reverted to runtime |
| Telegram pings silent | `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` secret missing or wrong | Composite action no-ops silently when unset; verify in repo secrets |
| Budget alerts never fire | Email recipient not wired in console | One-time step in Billing → Budgets → Manage notifications |
| `make destroy` doesn't drop spend to zero | AR images + GCS buckets accrue storage | Expected — €0.10/GB/mo for AR, €0.02/GB/mo for GCS. Delete manually if needed |

### Recovering from a half-bootstrapped state

`setup-op.sh` and `setup-iam.sh` are both idempotent — list-describe-create-or-update. Safe to re-run after a transient failure. The one thing that **isn't** idempotent: the budget email recipient (console-only).

---

## 5. Architecture decisions (why this shape)

| Decision | Why |
|---|---|
| **`garassino-op` holds WIF + secrets** | Single pool federated by all `ml` + `ai` projects (workspace 3-project model) |
| **GHCR for train image, GAR for runtime images** | Train image is pulled externally by RunPod (GHCR public is free); runtime images are pulled by Cloud Run in-region from GAR (faster cold start + cleaner IAM) |
| **CUDA *devel* base for inference** | SG3 custom ops at `deepVogue/pytorch_utils/ops/{bias_act,upfirdn2d}.py` JIT-compile via `torch.utils.cpp_extension.load()` at **import** time. Runtime base lacks nvcc → import fails |
| **CUDA devel for train, same reason** | Same custom ops, same JIT, same nvcc requirement |
| **Neon over Cloud SQL** | €0 vs €7/mo db-f1-micro. Neon auto-pauses after 5 min idle (~1s cold start); acceptable for show-and-destroy demo windows. Cross-project Secret Manager refs in the Cloud Run YAMLs |
| **Self-terminate via RunPod GraphQL** | RunPod pods don't auto-stop when the inner container exits — `desiredStatus` is the *requested* state, not actual. Entrypoint calls `podTerminate` on EXIT trap; orchestrator polls and treats `GONE`/`TERMINATED` as success |
| **MLflow IAP id-token refresher** | Cloud Run + IAP needs `Authorization: Bearer <id_token>`. MLflow's Python client reads `MLFLOW_TRACKING_TOKEN`. A daemon thread refreshes it every 30 min from the GCE metadata server. No-op without `IAP_OAUTH_CLIENT_ID` |
| **Trainer SA JSON key is the one exception to "no SA JSON keys"** | RunPod is external compute — can't federate via WIF. Stored in `op-trainer-key` Secret Manager, never committed |
| **`make show` / `make destroy` (not Terraform)** | Pragmatic v1 matches the existing imperative `setup-*.sh` style. TF state bucket already exists in `op` for v2 |
| **Cloud SQL CPU alert dropped from monitoring** | Neon has its own dashboards; Cloud Monitoring can't observe an external DB |

---

## 6. Cost expectations

Realistic monthly spend under normal use (one demo + one training run per week):

| Line item | Idle | Demo hour | Training run | Notes |
|---|---|---|---|---|
| Cloud Run (3 services, `minScale=0`) | €0 | ~€0.50 | — | scale-to-zero |
| Artifact Registry (5 images, ~5GB) | ~€0.50/mo | — | — | persistent storage |
| GCS buckets (datasets + models + logs, ~10GB) | ~€0.20/mo | — | ~€0.10 added | datasets dominate |
| Cloud Monitoring (uptime + 3 policies) | €0 | €0 | €0 | well under free tier |
| Neon free tier | €0 | €0 | €0 | 100 connections / 3GB cap |
| GHCR (public packages) | €0 | — | — | unlimited public |
| RunPod H100 80GB | — | — | ~€3.50/hr active | per-second billing; ~€0.20 for a 4-min smoke |
| RunPod A40 (alternative) | — | — | ~€0.40/hr | for cheap smoke runs |
| **Typical month** | **~€0.70** | + ~€2 | + ~€10 | well under €25 cap |

Budget alerts fire at 40% (€10), 80% (€20), 100% (€25) to `juan.garassino@hotmail.com`.

---

## 7. Where things live

```
deepVogue-mlops/
├── deepVogue/
│   ├── clients.py                    # fsspec + URI helpers (strip_scheme, split_uri, scheme_of)
│   ├── publish.py                    # snapshot → GCS + models.yaml upsert
│   ├── notifications/slack.py        # in-app slack pings (no-op when SLACK_WEBHOOK_URL unset)
│   ├── orchestration/
│   │   ├── flows.py                  # Prefect flow definitions (train/prepare/walk/...)
│   │   └── backends/
│   │       ├── local.py              # nano-stack mock + real prepare/publish
│   │       ├── colab.py              # v2 stub — manual notebook path for v1
│   │       ├── runpod.py             # real GPU pod submission + lifecycle
│   │       └── vertex.py             # Vertex AI CustomJob — GCP-native, no SA key
│   ├── serve/
│   │   ├── api.py                    # FastAPI inference (+ IAP refresher hook)
│   │   ├── loader.py                 # .pkl loader + GPU warmup
│   │   └── registry.py               # models.yaml reader
│   └── tracking/
│       ├── iap.py                    # GCE metadata id-token refresher
│       └── mlflow_helpers.py         # training-run logger
├── infra/
│   ├── docker/{train,inference,mlflow,prefect}/Dockerfile
│   ├── cloudrun/{inference,mlflow,prefect-server}.service.yaml
│   ├── cloudrun/prefect-worker.job.yaml
│   └── gcp/
│       ├── _common.sh                # PROJECT/REGION/OP_PROJECT, step, grant helpers
│       ├── setup.sh                  # ml: APIs + buckets + AR repo
│       ├── setup-op.sh               # op: WIF + TF state + log sink + secrets
│       ├── setup-iam.sh              # ml: runtime SAs + cross-project WIF binding
│       ├── setup-db-secrets.sh       # op: Neon DSN upsert + ml SA grants
│       ├── setup-monitoring.sh       # ml: uptime + alert policies + Slack channel
│       └── setup-budget.sh           # ml: €25 budget + 40/80/100% thresholds
├── scripts/
│   ├── submit_train.py               # CLI wrapper around backends.{runpod,vertex}.train
│   └── run_nano_smoke.py             # docker-compose end-to-end smoke
├── .github/
│   ├── actions/notify-telegram/      # composite — silent without TELEGRAM_* secrets
│   └── workflows/
│       ├── test.yml                  # black + pytest on push/PR
│       ├── build-{inference,mlflow,prefect}.yml  # auto-deploy to Cloud Run
│       ├── build-train.yml           # GHCR push + DV_FAKE_TRAIN smoke + visibility flip
│       └── nano-smoke.yml            # nightly docker-compose stack test
├── Makefile                          # canonical entry point for every operation
├── docs/OPERATIONS.md                # this file
├── CLAUDE.md                         # in-repo guidance for Claude Code sessions
└── README.md                         # public-facing intro + quickstart
```

---

## 8. The unfinished

These are deliberately out-of-scope for v1; carried forward as backlog:

- **Frontend** — separate workstream
- **`backends/colab.py` real handoff queue** — v2 design captured in the module docstring
- **Latent-cinema film pick** — creative decision
- **Terraform-isation** of imperative `setup-*.sh` — wait until op's TF state bucket holds real state
- **Per-tick MLflow logging from RunPod pods** — IAP id-token broker for non-GCP compute; v2

Everything else from earlier session standings is closed.
