#!/usr/bin/env bash
set -euo pipefail
PROJECT="${GCP_PROJECT:?}"
GH_REPO="${GITHUB_REPO:?owner/repo}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

# Runtime SAs
for sa in mlflow-server prefect-server prefect-worker fastapi-inference deepvogue-deployer; do
  if ! gcloud --project "$PROJECT" iam service-accounts describe \
       "${sa}-sa@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud --project "$PROJECT" iam service-accounts create "${sa}-sa" \
      --display-name="deepVogue $sa"
  fi
done

# Permissions — least-privilege
grant() {
  gcloud --project "$PROJECT" projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$1" --role="$2" --condition=None 2>/dev/null || true
}
grant "mlflow-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectAdmin
grant "mlflow-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "prefect-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "prefect-worker-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectAdmin
grant "prefect-worker-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "fastapi-inference-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectViewer
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/artifactregistry.writer
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/run.admin
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/iam.serviceAccountUser
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client

# WIF pool + provider for GitHub Actions
POOL="github-pool"
PROVIDER="github-provider"
if ! gcloud --project "$PROJECT" iam workload-identity-pools describe "$POOL" \
     --location=global >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools create "$POOL" --location=global
fi
if ! gcloud --project "$PROJECT" iam workload-identity-pools providers describe "$PROVIDER" \
     --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global --workload-identity-pool="$POOL" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="attribute.repository=='${GH_REPO}'"
fi

# Allow the GitHub repo to impersonate the deployer SA
gcloud --project "$PROJECT" iam service-accounts add-iam-policy-binding \
  "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GH_REPO}"

echo
echo "Add these as GitHub repo secrets:"
echo "  GCP_PROJECT=${PROJECT}"
echo "  GCP_WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo "  GCP_DEPLOYER_SA=deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com"
