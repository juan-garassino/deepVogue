#!/usr/bin/env bash
set -euo pipefail
PROJECT="${GCP_PROJECT:?}"
REGION="${GCP_REGION:-us-central1}"
INSTANCE="deepvogue-pg"
NETWORK="default"

if ! gcloud --project "$PROJECT" sql instances describe "$INSTANCE" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" sql instances create "$INSTANCE" \
    --database-version=POSTGRES_15 --tier=db-f1-micro --region="$REGION" \
    --network="$NETWORK" --no-assign-ip
fi

for db in mlflow prefect; do
  gcloud --project "$PROJECT" sql databases create "$db" --instance="$INSTANCE" 2>/dev/null || true
done

PASS="$(openssl rand -base64 24 | tr -d '/+=')"
gcloud --project "$PROJECT" sql users create dv --instance="$INSTANCE" --password="$PASS" 2>/dev/null || true
echo "Cloud SQL user 'dv' password: $PASS"
echo "Save this to Secret Manager: gcloud secrets create deepvogue-pg-password --data-file=- <<<$PASS"

# VPC connector for Cloud Run -> private SQL
if ! gcloud --project "$PROJECT" compute networks vpc-access connectors describe \
     deepvogue-vpc --region="$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" compute networks vpc-access connectors create deepvogue-vpc \
    --region="$REGION" --network="$NETWORK" --range=10.8.0.0/28
fi
