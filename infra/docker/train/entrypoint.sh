#!/usr/bin/env bash
# RunPod training entrypoint for deepVogue.
#
# Required env (forwarded by deepVogue/orchestration/backends/runpod.py):
#   DV_DATASET_URI    gs:// URI of dataset.zip
#   DV_RUN_URI        gs:// URI prefix where snapshots get mirrored
#   DV_MODEL_ID       string id used by publish + models.yaml
#   DV_CFG            stylegan3-t | stylegan3-r | stylegan2 (default stylegan3-t)
#   DV_KIMG           training length in kimg
#   DV_GAMMA          R1 gamma
#   DV_BATCH          total batch size
#   DV_RES            resolution (e.g. 256, 512)
#   DV_PUBLISH_TARGET gs:// root where models.yaml lives
#   GOOGLE_APPLICATION_CREDENTIALS_JSON  full SA JSON (RunPod only); when
#                     absent, ambient ADC is used (Vertex AI / GCE metadata)
#   RUNPOD_API_KEY    used to self-terminate via GraphQL at the end (RunPod
#                     only; Vertex releases the VM itself)
#
# RunPod injects automatically:
#   RUNPOD_POD_ID     this pod's id (used for self-terminate)
#
# Optional:
#   DV_RESUME_FROM    gs:// URI of a network-snapshot-*.pkl to resume from
#   DV_FAKE_TRAIN     if set to 1, skip GPU code + emit stub pkl (CI smoke)
#   SYNC_INTERVAL     seconds between rsync ticks (default 60)
#   MLFLOW_TRACKING_URI / SLACK_WEBHOOK_URL passed through

set -euo pipefail

WORKDIR=/workspace
RUNDIR="$WORKDIR/run"
DATASET="$WORKDIR/dataset.zip"
mkdir -p "$RUNDIR"

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

require() {
    local var="$1"
    if [ -z "${!var:-}" ]; then
        log "ERROR: $var is required"
        exit 2
    fi
}

self_terminate() {
    # Best-effort: terminate this pod via the RunPod GraphQL API.
    # Retries 3x; never blocks the exit path.
    local pod_id="${RUNPOD_POD_ID:-}"
    local key="${RUNPOD_API_KEY:-}"
    if [ -z "$pod_id" ] || [ -z "$key" ]; then
        log "self-terminate: skipping (RUNPOD_POD_ID or RUNPOD_API_KEY unset)"
        return 0
    fi
    log "self-terminate: terminating pod $pod_id"
    local q='{"query":"mutation { podTerminate(input: {podId: \"'"$pod_id"'\"}) }"}'
    for i in 1 2 3; do
        if curl -fsS -X POST "https://api.runpod.io/graphql?api_key=${key}" \
            -H "Content-Type: application/json" -d "$q" >/dev/null 2>&1; then
            log "self-terminate: ok"
            return 0
        fi
        log "self-terminate: attempt $i failed, retrying"
        sleep 5
    done
    log "self-terminate: gave up; orchestrator will reap on timeout"
    return 0
}

# Run self-terminate on any exit path (success, error, signal).
trap 'self_terminate' EXIT

# ---------- 0. fake-train short-circuit (CI smoke; no GPU, no GCS) ----------
if [ "${DV_FAKE_TRAIN:-0}" = "1" ]; then
    require DV_KIMG
    log "DV_FAKE_TRAIN=1 -> emitting stub snapshot, skipping GPU + gsutil"
    python - <<'PY'
import os, shutil
from pathlib import Path
fixture = Path("tests/fixtures/stub_sg3_state_dict.pt")
out = Path("/workspace/run")
out.mkdir(parents=True, exist_ok=True)
kimg = int(os.environ["DV_KIMG"])
shutil.copy(fixture, out / f"network-snapshot-{kimg:06d}.pkl")
print(f"wrote stub snapshot at kimg={kimg}")
PY
    log "fake-train done."
    exit 0
fi

# ---------- 1. validate real-run env ----------
require DV_DATASET_URI
require DV_RUN_URI
require DV_MODEL_ID
require DV_KIMG
require DV_GAMMA
require DV_BATCH

: "${DV_CFG:=stylegan3-t}"
: "${DV_RES:=256}"
: "${SYNC_INTERVAL:=60}"

# ---------- 2. auth so gsutil works ----------
# RunPod (external compute): SA JSON key injected via env — the documented
# WIF exception. Vertex AI / GCE: no key needed; the attached service account
# provides ambient ADC through the metadata server.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
    SA_JSON=/tmp/sa.json
    printf '%s' "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > "$SA_JSON"
    chmod 600 "$SA_JSON"
    gcloud auth activate-service-account --key-file="$SA_JSON" --quiet
    export GOOGLE_APPLICATION_CREDENTIALS="$SA_JSON"
    log "gcloud auth: SA key — $(gcloud config get-value account 2>/dev/null)"
else
    log "gcloud auth: no SA key in env — using ambient ADC (Vertex/GCE metadata)"
fi

# ---------- 3. GPU + custom-ops warmup (fail fast on CUDA mismatch) ----------
nvidia-smi || { log "ERROR: nvidia-smi unavailable"; exit 3; }
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
python - <<'PY'
import torch
from deepVogue.pytorch_utils.ops import bias_act, upfirdn2d
x = torch.zeros(1, 1, 4, 4, device="cuda")
bias_act.bias_act(x)
upfirdn2d.upfirdn2d(x, torch.ones(1, 1, device="cuda"))
print("custom ops OK")
PY

# ---------- 4. pull dataset ----------
log "fetching dataset $DV_DATASET_URI -> $DATASET"
gsutil -q -m cp "$DV_DATASET_URI" "$DATASET"

RESUME_ARGS=()
if [ -n "${DV_RESUME_FROM:-}" ]; then
    log "fetching resume checkpoint $DV_RESUME_FROM"
    gsutil -q cp "$DV_RESUME_FROM" "$WORKDIR/resume.pkl"
    RESUME_ARGS=(--resume="$WORKDIR/resume.pkl")
fi

# ---------- 5. background snapshot mirror ----------
mirror_loop() {
    while sleep "$SYNC_INTERVAL"; do
        gsutil -q -m rsync -r "$RUNDIR" "$DV_RUN_URI" || log "rsync tick failed (continuing)"
    done
}
mirror_loop &
MIRROR_PID=$!

# ---------- 6. train ----------
log "starting train.py cfg=$DV_CFG res=$DV_RES kimg=$DV_KIMG gamma=$DV_GAMMA batch=$DV_BATCH"
python deepVogue/train.py \
    --outdir="$RUNDIR" \
    --data="$DATASET" \
    --cfg="$DV_CFG" \
    --gpus=1 \
    --kimg="$DV_KIMG" \
    --gamma="$DV_GAMMA" \
    --batch="$DV_BATCH" \
    --metrics=fid50k_full \
    "${RESUME_ARGS[@]}"

# ---------- 7. final mirror + publish ----------
kill "$MIRROR_PID" 2>/dev/null || true
log "final rsync $RUNDIR -> $DV_RUN_URI"
gsutil -q -m rsync -r "$RUNDIR" "$DV_RUN_URI"

if [ -n "${DV_PUBLISH_TARGET:-}" ]; then
    log "publishing $DV_MODEL_ID -> $DV_PUBLISH_TARGET"
    python -m deepVogue.publish \
        --model-id="$DV_MODEL_ID" \
        --src-dir="$RUNDIR"
else
    log "DV_PUBLISH_TARGET unset; skipping publish step"
fi

log "done."
