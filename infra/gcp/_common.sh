#!/usr/bin/env bash
# Shared bootstrap for infra/gcp/*.sh — source from each setup script.
# Sets strict mode and resolves the two env vars used by every step.
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-europe-west1}"
OP_PROJECT="${GCP_OP_PROJECT:-garassino-op}"

step() { echo; echo "==> $*"; }

# Resolve garassino-op's project number once (used for cross-project WIF binding).
resolve_op_number() {
  gcloud projects describe "$OP_PROJECT" --format='value(projectNumber)' 2>/dev/null
}
