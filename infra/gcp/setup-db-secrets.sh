#!/usr/bin/env bash
# Stash Neon DSNs into garassino-op's Secret Manager and grant read access to
# the MLflow + Prefect Cloud Run runtime SAs in garassino-ml.
#
# Required env (when actually writing): NEON_MLFLOW_DSN, NEON_PREFECT_DSN.
# Skips with a warning if the DSNs aren't set — re-run via `make deploy-db-secrets`
# after you've minted the Neon project.

# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

OP_NUMBER=$(resolve_op_number)
if [ -z "$OP_NUMBER" ]; then
  echo "ERROR: cannot resolve $OP_PROJECT — run \`make gcp-setup-op\` first" >&2
  exit 1
fi

if [ -z "${NEON_MLFLOW_DSN:-}" ] || [ -z "${NEON_PREFECT_DSN:-}" ]; then
  echo "WARN: NEON_MLFLOW_DSN or NEON_PREFECT_DSN unset"
  echo "      mint a Neon project at neon.tech, then re-run \`make deploy-db-secrets\` with both vars set"
  exit 0
fi

upsert_secret() {
  local name="$1" value="$2"
  if ! gcloud --project "$OP_PROJECT" secrets describe "$name" >/dev/null 2>&1; then
    gcloud --project "$OP_PROJECT" secrets create "$name" --replication-policy=automatic
  fi
  printf '%s' "$value" | gcloud --project "$OP_PROJECT" secrets versions add "$name" --data-file=-
  echo "  upserted: ${OP_PROJECT}/${name}"
}

step "Push Neon DSNs into ${OP_PROJECT}'s Secret Manager"
upsert_secret op-neon-mlflow-dsn "$NEON_MLFLOW_DSN"
upsert_secret op-neon-prefect-dsn "$NEON_PREFECT_DSN"

step "Grant secretAccessor on op secrets to ml's runtime SAs"
grant_access() {
  local sa="$1" secret="$2"
  gcloud --project "$OP_PROJECT" secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:$sa" \
    --role=roles/secretmanager.secretAccessor \
    --condition=None 2>/dev/null || true
  echo "  ${sa} -> ${secret}"
}
grant_access "mlflow-server-sa@${PROJECT}.iam.gserviceaccount.com" op-neon-mlflow-dsn
grant_access "prefect-server-sa@${PROJECT}.iam.gserviceaccount.com" op-neon-prefect-dsn
grant_access "prefect-worker-sa@${PROJECT}.iam.gserviceaccount.com" op-neon-prefect-dsn

echo
echo "Cloud Run YAMLs already wire these in via valueFrom.secretKeyRef."
echo "Cleanup: if a legacy \`deepvogue-pg\` Cloud SQL instance exists, delete it manually:"
echo "  gcloud sql instances delete deepvogue-pg --project=${PROJECT}"
