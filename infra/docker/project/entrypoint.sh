#!/usr/bin/env bash
# Cloud Run L4 projection entrypoint — latent cinema, phase 1 (anchors).
#
# Projects every DV_PROJ_STRIDE-th film frame into the generator's W+ space
# (deepVogue.cinema project-frames) so each frame becomes an anchor for the
# ordered latent walk. Slice-friendly: project-frames skips frames whose
# projected_w.npz already exists, and anchors mirror to GCS every tick, so a
# 3600s Cloud Run cap just means the next slice resumes where this one stopped.
#
# Required env:
#   DV_DATASET_URI        gs:// dataset.zip (frames live inside it)
#   DV_FRAMES_INDEX_URI   gs:// frames_index.json (temporal order + filenames)
#   DV_NETWORK_URI        gs:// network-snapshot-*.pkl to project against
#   DV_ANCHORS_URI        gs:// prefix where projected_w.npz anchors mirror
# Optional:
#   DV_PROJ_STRIDE        project every Nth frame (default 16)
#   DV_PROJ_STEPS         projector optimization steps/frame (default 250)
#   SYNC_INTERVAL         anchor-mirror period in seconds (default 120)
set -euo pipefail

WORKDIR=/workspace
export DV_DATASET_DIR="$WORKDIR/dataset"
export DV_ANCHORS_DIR="$WORKDIR/anchors"
export DV_NETWORK_PKL="$WORKDIR/network.pkl"
unset DV_DATASET_NAME 2>/dev/null || true   # explicit dirs; no name-suffixing
mkdir -p "$DV_DATASET_DIR" "$DV_ANCHORS_DIR"

log() { printf '[project %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
require() { [ -n "${!1:-}" ] || { log "ERROR: $1 is required"; exit 2; }; }

require DV_DATASET_URI
require DV_FRAMES_INDEX_URI
require DV_NETWORK_URI
require DV_ANCHORS_URI
: "${DV_PROJ_STRIDE:=16}"
: "${DV_PROJ_STEPS:=250}"
: "${SYNC_INTERVAL:=120}"

# auth (ambient ADC on Cloud Run; SA-key path kept for parity with train)
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
    printf '%s' "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/sa.json && chmod 600 /tmp/sa.json
    gcloud auth activate-service-account --key-file=/tmp/sa.json --quiet
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa.json
else
    log "gcloud auth: ambient ADC"
fi

nvidia-smi || { log "ERROR: no GPU"; exit 3; }

log "fetching inputs"
gsutil -q cp "$DV_DATASET_URI" "$DV_DATASET_DIR/dataset.zip"
gsutil -q cp "$DV_FRAMES_INDEX_URI" "$DV_DATASET_DIR/frames_index.json"
gsutil -q cp "$DV_NETWORK_URI" "$DV_NETWORK_PKL"
# resume: pull anchors already projected by earlier slices
gsutil -q -m rsync -r "$DV_ANCHORS_URI" "$DV_ANCHORS_DIR" 2>/dev/null || true

mirror_loop() {
    while sleep "$SYNC_INTERVAL"; do
        gsutil -q -m rsync -r "$DV_ANCHORS_DIR" "$DV_ANCHORS_URI" || log "anchor rsync tick failed (continuing)"
    done
}
mirror_loop & MIRROR_PID=$!

log "projecting stride=$DV_PROJ_STRIDE steps=$DV_PROJ_STEPS network=$DV_NETWORK_URI"
python -m deepVogue.cinema project-frames --stride "$DV_PROJ_STRIDE" --num-steps "$DV_PROJ_STEPS"

kill "$MIRROR_PID" 2>/dev/null || true
log "final anchor rsync -> $DV_ANCHORS_URI"
gsutil -q -m rsync -r "$DV_ANCHORS_DIR" "$DV_ANCHORS_URI"
log "done."
