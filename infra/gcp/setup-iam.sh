#!/usr/bin/env bash
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
GH_REPO="${GITHUB_REPO:?owner/repo}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

# Runtime SAs
for sa in mlflow-server prefect-server prefect-worker fastapi-inference deepvogue-deployer deepvogue-trainer; do
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
grant "deepvogue-trainer-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectAdmin
# Vertex AI training: the deployer submits CustomJobs that run AS the trainer SA;
# the Vertex service agent pulls the train image from Artifact Registry.
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/aiplatform.user
gcloud --project "$PROJECT" iam service-accounts add-iam-policy-binding \
  "deepvogue-trainer-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --member="serviceAccount:deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" \
  --condition=None 2>/dev/null || true
grant "service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" roles/artifactregistry.reader
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/artifactregistry.writer
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/run.admin
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/iam.serviceAccountUser
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client

# Cross-project WIF: garassino-op owns the github-pool / github-provider.
# This script only adds a binding so the deployer SA in $PROJECT (garassino-ml)
# can be impersonated by GitHub Actions in the repo named $GH_REPO.
step "Cross-project WIF binding from $OP_PROJECT"
OP_NUMBER=$(resolve_op_number)
if [ -z "$OP_NUMBER" ]; then
  echo "ERROR: cannot resolve $OP_PROJECT — make sure garassino-op is bootstrapped first" >&2
  exit 1
fi
POOL="github-pool"
PROVIDER="github-provider"
gcloud --project "$PROJECT" iam service-accounts add-iam-policy-binding \
  "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${OP_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GH_REPO}"

echo
echo "Add these as GitHub repo secrets:"
echo "  GCP_PROJECT=${PROJECT}"
echo "  GCP_WIF_PROVIDER=projects/${OP_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo "  GCP_DEPLOYER_SA=deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com"
echo
# DEVIATION from the workspace-wide "no SA JSON keys anywhere" rule:
# RunPod is external compute → can't use Workload Identity Federation directly.
# The trainer key is the documented exception. Store it in Secret Manager
# (op/deepvogue-trainer-key) rather than checking it in.
echo "To mint a JSON key for the RunPod trainer SA (documented WIF exception — external compute):"
echo "  gcloud iam service-accounts keys create trainer-key.json \\"
echo "    --iam-account=deepvogue-trainer-sa@${PROJECT}.iam.gserviceaccount.com"
echo "Then set on the launcher: export GOOGLE_APPLICATION_CREDENTIALS_JSON=\"\$(cat trainer-key.json)\""
