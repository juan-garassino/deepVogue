#!/usr/bin/env bash
# Bootstrap garassino-op — the workspace control plane.
#
# Owns: WIF pool/provider for GitHub Actions, Terraform state bucket,
#       Secret Manager placeholders, centralized log sink for ml + ai.
#
# Run once: GCP_PROJECT=garassino-op GITHUB_REPO=owner/repo make gcp-setup-op
#
# Idempotent — re-running creates only what's missing.

set -euo pipefail

PROJECT="${GCP_PROJECT:-${GCP_OP_PROJECT:-garassino-op}}"
REGION="${GCP_REGION:-europe-west1}"
GH_REPO="${GITHUB_REPO:?set GITHUB_REPO=owner/repo for the WIF attribute condition}"

step() { echo; echo "==> $*"; }

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

# ---------------------------------------------------------------------------
# 1. WIF pool + provider for GitHub Actions (federated by garassino-ml + ai)
# ---------------------------------------------------------------------------
step "WIF pool + provider for $GH_REPO"
POOL="github-pool"
PROVIDER="github-provider"
if ! gcloud --project "$PROJECT" iam workload-identity-pools describe "$POOL" \
     --location=global >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools create "$POOL" --location=global
  echo "created pool: $POOL"
else
  echo "reusing pool: $POOL"
fi
if ! gcloud --project "$PROJECT" iam workload-identity-pools providers describe "$PROVIDER" \
     --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global --workload-identity-pool="$POOL" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="attribute.repository=='${GH_REPO}'"
  echo "created provider: $PROVIDER"
else
  echo "reusing provider: $PROVIDER"
fi

# ---------------------------------------------------------------------------
# 2. Terraform state bucket
# ---------------------------------------------------------------------------
step "TF state bucket"
TF_BUCKET="garassino-op-tf-state"
if ! gsutil ls -b "gs://${TF_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -p "$PROJECT" -l "$REGION" -b on "gs://${TF_BUCKET}"
  gsutil versioning set on "gs://${TF_BUCKET}"
  echo "created gs://${TF_BUCKET}"
else
  echo "reusing gs://${TF_BUCKET}"
fi

# ---------------------------------------------------------------------------
# 3. Centralized log sink (warnings+ from ml + ai → op-logs bucket)
# ---------------------------------------------------------------------------
step "Log sink + bucket"
LOGS_BUCKET="garassino-op-logs"
if ! gsutil ls -b "gs://${LOGS_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -p "$PROJECT" -l "$REGION" -b on "gs://${LOGS_BUCKET}"
  echo "created gs://${LOGS_BUCKET}"
fi

SINK_NAME="ml-ai-warnings"
SINK_DEST="storage.googleapis.com/${LOGS_BUCKET}"
SINK_FILTER='severity>=WARNING'
for SRC in garassino-ml garassino-ai; do
  if gcloud projects describe "$SRC" >/dev/null 2>&1; then
    if ! gcloud --project "$SRC" logging sinks describe "$SINK_NAME" >/dev/null 2>&1; then
      gcloud --project "$SRC" logging sinks create "$SINK_NAME" "$SINK_DEST" --log-filter="$SINK_FILTER"
      echo "  sink created on $SRC"
    else
      echo "  sink exists on $SRC"
    fi
    # Grant the sink's writer the right to write into the bucket.
    WRITER=$(gcloud --project "$SRC" logging sinks describe "$SINK_NAME" --format='value(writerIdentity)')
    gsutil iam ch "${WRITER}:roles/storage.objectCreator" "gs://${LOGS_BUCKET}" >/dev/null 2>&1 || true
  else
    echo "  $SRC not bootstrapped yet — skipping sink for now"
  fi
done

# ---------------------------------------------------------------------------
# 4. Secret Manager placeholders (empty; user fills via console / CLI)
# ---------------------------------------------------------------------------
step "Secret Manager placeholders"
for sec in op-openai-key op-anthropic-key op-slack-webhook op-runpod-api-key \
           op-trainer-key op-neon-mlflow-dsn op-neon-prefect-dsn; do
  if ! gcloud --project "$PROJECT" secrets describe "$sec" >/dev/null 2>&1; then
    gcloud --project "$PROJECT" secrets create "$sec" --replication-policy=automatic
    # Seed with an empty version so secretAccessor grants don't fail on first read.
    printf '' | gcloud --project "$PROJECT" secrets versions add "$sec" --data-file=-
    echo "  created secret: $sec"
  else
    echo "  reusing secret: $sec"
  fi
done

# ---------------------------------------------------------------------------
# 5. Echo block — GitHub repo secrets for downstream `garassino-ml` setup
# ---------------------------------------------------------------------------
echo
echo "Add these as GitHub repo secrets (used by garassino-ml deploys):"
echo "  GCP_WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo
echo "Next step: run \`GCP_PROJECT=garassino-ml GITHUB_REPO=${GH_REPO} make gcp-setup\`"
