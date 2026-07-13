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
#   DV_AUTO_RESUME    if 1 (Cloud Run slice mode): mirror into a per-slice
#                     subdir of DV_RUN_URI and resume from the latest snapshot
#                     across all previous slices; DV_RESUME_FROM is only the
#                     cold-start fallback (e.g. a pretrained pkl)
#   DV_TARGET_KIMG    unattended-chain cost cap: in auto-resume mode, no-op
#                     exit once cumulative kimg (completed slices × DV_KIMG)
#                     reaches this. Unset on manual one-off slices.
#   DV_METRICS        train.py --metrics (default fid50k_full; use "none" on
#                     1h slices — fid50k eats most of a slice)
#   DV_SNAP           train.py --snap in ticks (default 50; use 2-4 on slices)
#   DV_CBASE          train.py --cbase; MUST match the resume pkl's capacity
#                     (NVIDIA's *-256x256 pretrained pkls use 16384, train.py
#                     defaults to 32768 — mismatch fails net construction)
#   DV_CMAX           train.py --cmax (default 512)
#   DV_MIRROR         train.py --mirror (dataset x-flips; default false)
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

# ---------- 2.5 target-kimg self-limit (unattended-chain cost cap) ----------
# When a Cloud Scheduler fires slices unattended, this is the hard stop:
# count completed slices (each mirrors one final network-snapshot-<DV_KIMG>.pkl)
# and no-op exit once cumulative kimg reaches DV_TARGET_KIMG. Runs before GPU
# warmup + the 1 GB dataset pull so an over-target fire is a ~seconds no-op.
# Only active in auto-resume (chain) mode; a one-off manual slice ignores it.
if [ -n "${DV_TARGET_KIMG:-}" ] && [ "${DV_AUTO_RESUME:-0}" = "1" ]; then
    FINAL_SNAP=$(printf 'network-snapshot-%06d.pkl' "$DV_KIMG")
    DONE=$(gsutil ls "${DV_RUN_URI%/}/slices/**/${FINAL_SNAP}" 2>/dev/null | grep -c . || true)
    CUM=$(( DONE * DV_KIMG ))
    if [ "$CUM" -ge "$DV_TARGET_KIMG" ]; then
        log "target reached: ~${CUM} kimg done (>= DV_TARGET_KIMG=${DV_TARGET_KIMG}); no-op exit"
        exit 0
    fi
    log "chain progress: ~${CUM}/${DV_TARGET_KIMG} kimg done; training this slice"
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

# Slice mode (Cloud Run 1h GPU jobs): train.py restarts run-dir numbering at
# 00000- every container, so consecutive slices rsynced into one prefix would
# overwrite each other's snapshots. Give each slice its own subdir and resume
# from the lexically-last snapshot across all of them (timestamped dirs sort
# chronologically, snapshot names sort by kimg within a dir).
MIRROR_URI="$DV_RUN_URI"
RESUME_SRC="${DV_RESUME_FROM:-}"
if [ "${DV_AUTO_RESUME:-0}" = "1" ]; then
    MIRROR_URI="${DV_RUN_URI%/}/slices/$(date -u +%Y%m%dT%H%M%SZ)"
    LATEST_SNAP=$(gsutil ls "${DV_RUN_URI%/}/slices/**network-snapshot-*.pkl" 2>/dev/null | sort | tail -1 || true)
    if [ -n "$LATEST_SNAP" ]; then
        log "auto-resume: latest slice snapshot is $LATEST_SNAP"
        RESUME_SRC="$LATEST_SNAP"
    else
        log "auto-resume: no prior slice snapshots; falling back to DV_RESUME_FROM=${RESUME_SRC:-<unset>}"
    fi
fi

RESUME_ARGS=()
if [ -n "$RESUME_SRC" ]; then
    log "fetching resume checkpoint $RESUME_SRC"
    gsutil -q cp "$RESUME_SRC" "$WORKDIR/resume.pkl"
    RESUME_ARGS=(--resume="$WORKDIR/resume.pkl")
fi

# ---------- 5. background snapshot mirror ----------
mirror_loop() {
    while sleep "$SYNC_INTERVAL"; do
        gsutil -q -m rsync -r "$RUNDIR" "$MIRROR_URI" || log "rsync tick failed (continuing)"
    done
}
mirror_loop &
MIRROR_PID=$!

# ---------- 6. train ----------
EXTRA_ARGS=()
[ -n "${DV_CBASE:-}" ] && EXTRA_ARGS+=(--cbase="$DV_CBASE")
[ -n "${DV_CMAX:-}" ] && EXTRA_ARGS+=(--cmax="$DV_CMAX")
[ -n "${DV_MIRROR:-}" ] && EXTRA_ARGS+=(--mirror="$DV_MIRROR")

log "starting train.py cfg=$DV_CFG res=$DV_RES kimg=$DV_KIMG gamma=$DV_GAMMA batch=$DV_BATCH extra=${EXTRA_ARGS[*]:-none}"
python deepVogue/train.py \
    --outdir="$RUNDIR" \
    --data="$DATASET" \
    --cfg="$DV_CFG" \
    --gpus=1 \
    --kimg="$DV_KIMG" \
    --gamma="$DV_GAMMA" \
    --batch="$DV_BATCH" \
    --metrics="${DV_METRICS:-fid50k_full}" \
    --snap="${DV_SNAP:-50}" \
    "${EXTRA_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

# ---------- 7. final mirror + publish ----------
kill "$MIRROR_PID" 2>/dev/null || true
log "final rsync $RUNDIR -> $MIRROR_URI"
gsutil -q -m rsync -r "$RUNDIR" "$MIRROR_URI"

if [ -n "${DV_PUBLISH_TARGET:-}" ]; then
    log "publishing $DV_MODEL_ID -> $DV_PUBLISH_TARGET"
    python -m deepVogue.publish \
        --model-id="$DV_MODEL_ID" \
        --src-dir="$RUNDIR"
else
    log "DV_PUBLISH_TARGET unset; skipping publish step"
fi

log "done."
