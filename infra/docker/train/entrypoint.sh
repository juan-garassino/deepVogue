#!/usr/bin/env bash
# RunPod training entrypoint for deepVogue.
#
# Required env:
#   DV_DATASET_URI    gs:// URI of dataset.zip
#   DV_RUN_URI        gs:// URI prefix where snapshots get mirrored
#   DV_MODEL_ID       string id used by publish + models.yaml
#   DV_CFG            stylegan3-t | stylegan3-r | stylegan2 (default stylegan3-t)
#   DV_KIMG           training length in kimg
#   DV_GAMMA          R1 gamma
#   DV_BATCH          total batch size
#   DV_RES            resolution (e.g. 256, 512)
#   DV_PUBLISH_TARGET gs:// root where models.yaml lives (used by publish step)
#   MLFLOW_TRACKING_URI  passed through to train.py / publish for run logging
#
# Optional:
#   DV_RESUME_FROM    gs:// URI of a network-snapshot-*.pkl to resume from
#   DV_FAKE_TRAIN     if set to 1, skip GPU code and emit stub pkl (CI smoke)
#   SYNC_INTERVAL     seconds between rsync ticks (default 60)
#   SLACK_WEBHOOK_URL forwarded to publish/notify

set -euo pipefail

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

require() {
    local var="$1"
    if [ -z "${!var:-}" ]; then
        log "ERROR: $var is required"
        exit 2
    fi
}

require DV_DATASET_URI
require DV_RUN_URI
require DV_MODEL_ID
require DV_KIMG
require DV_GAMMA
require DV_BATCH

: "${DV_CFG:=stylegan3-t}"
: "${DV_RES:=256}"
: "${SYNC_INTERVAL:=60}"

WORKDIR=/workspace
RUNDIR="$WORKDIR/run"
DATASET="$WORKDIR/dataset.zip"
mkdir -p "$RUNDIR"

# ---------- 1. pull dataset ----------
log "fetching dataset $DV_DATASET_URI -> $DATASET"
gsutil -q -m cp "$DV_DATASET_URI" "$DATASET"

RESUME_ARGS=()
if [ -n "${DV_RESUME_FROM:-}" ]; then
    log "fetching resume checkpoint $DV_RESUME_FROM"
    gsutil -q cp "$DV_RESUME_FROM" "$WORKDIR/resume.pkl"
    RESUME_ARGS=(--resume="$WORKDIR/resume.pkl")
fi

# ---------- 2. background snapshot mirror ----------
mirror_loop() {
    while sleep "$SYNC_INTERVAL"; do
        gsutil -q -m rsync -r "$RUNDIR" "$DV_RUN_URI" || log "rsync tick failed (continuing)"
    done
}
mirror_loop &
MIRROR_PID=$!
trap 'kill "$MIRROR_PID" 2>/dev/null || true' EXIT

# ---------- 3. train ----------
if [ "${DV_FAKE_TRAIN:-0}" = "1" ]; then
    log "DV_FAKE_TRAIN=1 -> emitting stub snapshot, skipping GPU training"
    python - <<'PY'
import os, shutil
from pathlib import Path
fixture = Path("tests/fixtures/stub_sg3_state_dict.pt")
out = Path(os.environ["RUNDIR"] if "RUNDIR" in os.environ else "/workspace/run")
out.mkdir(parents=True, exist_ok=True)
kimg = int(os.environ["DV_KIMG"])
shutil.copy(fixture, out / f"network-snapshot-{kimg:06d}.pkl")
PY
else
    log "starting deepVogue/train.py cfg=$DV_CFG res=$DV_RES kimg=$DV_KIMG gamma=$DV_GAMMA batch=$DV_BATCH"
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
fi

# ---------- 4. final mirror + publish ----------
log "final rsync $RUNDIR -> $DV_RUN_URI"
gsutil -q -m rsync -r "$RUNDIR" "$DV_RUN_URI"

if [ -n "${DV_PUBLISH_TARGET:-}" ]; then
    log "publishing $DV_MODEL_ID"
    python -m deepVogue.publish \
        --model-id="$DV_MODEL_ID" \
        --src-dir="$RUNDIR"
else
    log "DV_PUBLISH_TARGET unset; skipping publish step"
fi

log "done."
