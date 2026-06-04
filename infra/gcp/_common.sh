#!/usr/bin/env bash
# Shared bootstrap for infra/gcp/*.sh — source from each setup script.
# Sets strict mode and resolves the two env vars used by every step.
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"

step() { echo; echo "==> $*"; }
