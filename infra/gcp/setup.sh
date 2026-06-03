#!/usr/bin/env bash
# Idempotent GCP bootstrap for deepVogue MLOps stack.
# Required env: GCP_PROJECT, GCP_REGION (default us-central1), GITHUB_REPO (owner/repo)
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
GH_REPO="${GITHUB_REPO:?set GITHUB_REPO=owner/repo}"

step() { echo; echo "==> $*"; }

step "Enable APIs"
gcloud --project "$PROJECT" services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com \
  iam.googleapis.com \
  vpcaccess.googleapis.com \
  servicenetworking.googleapis.com \
  iap.googleapis.com

step "GCS buckets"
for b in models datasets walks mlflow queue; do
  if ! gcloud --project "$PROJECT" storage buckets describe "gs://deepvogue-$b" >/dev/null 2>&1; then
    gcloud --project "$PROJECT" storage buckets create "gs://deepvogue-$b" \
      --location="$REGION" --uniform-bucket-level-access
  fi
done

step "Artifact Registry repo"
if ! gcloud --project "$PROJECT" artifacts repositories describe deepvogue \
     --location="$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" artifacts repositories create deepvogue \
    --repository-format=docker --location="$REGION" --description="deepVogue images"
fi

echo
echo "Stage 1 (APIs, buckets, AR) complete."
echo "Next: infra/gcp/setup-sql.sh and infra/gcp/setup-iam.sh"
