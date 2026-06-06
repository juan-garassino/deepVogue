#!/usr/bin/env bash
# Cloud Monitoring uptime checks + alert policies for the deepVogue stack.
#
# Idempotent: re-running upserts policies + uptime checks. Notifications go to a
# Slack channel resource created from SLACK_WEBHOOK_URL (set in env). If the
# webhook is unset, the script still creates uptime checks + policies but they
# will fire with no destination.
#
# Required env: GCP_PROJECT, GCP_REGION (default us-central1).
# Optional env: SLACK_WEBHOOK_URL, SLACK_CHANNEL (default '#alerts').

# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

SLACK_CHANNEL_DISPLAY="${SLACK_CHANNEL:-#alerts}"

# ---------------------------------------------------------------------------
# 1. Slack notification channel
# ---------------------------------------------------------------------------
step "Notification channel (Slack)"
CHANNEL_NAME=""
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  # Look up by display name first (idempotent re-run); create if missing.
  CHANNEL_NAME=$(gcloud --project "$PROJECT" alpha monitoring channels list \
    --filter="displayName=\"deepvogue-slack\"" \
    --format="value(name)" 2>/dev/null | head -n1 || true)
  if [ -z "$CHANNEL_NAME" ]; then
    CHANNEL_NAME=$(gcloud --project "$PROJECT" alpha monitoring channels create \
      --display-name="deepvogue-slack" \
      --type=webhook_tokenauth \
      --channel-labels="url=${SLACK_WEBHOOK_URL}" \
      --format="value(name)")
    echo "created channel: $CHANNEL_NAME"
  else
    echo "reusing channel: $CHANNEL_NAME"
  fi
else
  echo "SLACK_WEBHOOK_URL unset; skipping channel creation. Policies will be created without notifications."
fi

# ---------------------------------------------------------------------------
# 2. Uptime checks against /health on each Cloud Run service
# ---------------------------------------------------------------------------
upsert_uptime() {
  local svc="$1"
  local host
  host=$(gcloud --project "$PROJECT" run services describe "$svc" \
    --region="$REGION" --format='value(status.url)' 2>/dev/null \
    | sed -E 's#^https?://##; s#/.*$##') || true
  if [ -z "$host" ]; then
    echo "  service $svc not deployed yet; skipping uptime"
    return 0
  fi
  local display="uptime-${svc}"
  # Check if uptime already exists by display name.
  if gcloud --project "$PROJECT" monitoring uptime list-configs \
      --format="value(displayName)" 2>/dev/null | grep -qx "$display"; then
    echo "  uptime $display exists; leaving alone"
    return 0
  fi
  gcloud --project "$PROJECT" monitoring uptime create "$display" \
    --resource-type=uptime-url \
    --resource-labels="host=${host},project_id=${PROJECT}" \
    --path="/health" \
    --port=443 \
    --protocol=https \
    --period=5m \
    --timeout=10s
  echo "  created uptime: $display (host=$host)"
}

step "Uptime checks"
for svc in deepvogue-inference deepvogue-mlflow deepvogue-prefect-server; do
  upsert_uptime "$svc"
done

# ---------------------------------------------------------------------------
# 3. Alert policies (uptime fail + Cloud Run 5xx + Cloud SQL CPU)
# ---------------------------------------------------------------------------
step "Alert policies"
POLICY_TMP="$(mktemp -d)"
trap 'rm -rf "$POLICY_TMP"' EXIT

write_policy() {
  local name="$1" body="$2"
  echo "$body" > "$POLICY_TMP/${name}.yaml"
  local existing
  existing=$(gcloud --project "$PROJECT" alpha monitoring policies list \
    --filter="displayName=\"${name}\"" --format="value(name)" 2>/dev/null | head -n1 || true)
  if [ -n "$existing" ]; then
    gcloud --project "$PROJECT" alpha monitoring policies update "$existing" \
      --policy-from-file="$POLICY_TMP/${name}.yaml" --quiet || true
    echo "  updated policy: $name"
  else
    local args=(--policy-from-file="$POLICY_TMP/${name}.yaml")
    [ -n "$CHANNEL_NAME" ] && args+=(--notification-channels="$CHANNEL_NAME")
    gcloud --project "$PROJECT" alpha monitoring policies create "${args[@]}"
    echo "  created policy: $name"
  fi
}

# Cloud Run 5xx ratio > 5% over 10 min on inference.
write_policy "deepvogue-inference-5xx" "$(cat <<'YAML'
displayName: deepvogue-inference-5xx
combiner: OR
conditions:
- displayName: 5xx rate > 5% (10m)
  conditionThreshold:
    filter: |
      metric.type="run.googleapis.com/request_count"
      resource.type="cloud_run_revision"
      resource.labels.service_name="deepvogue-inference"
      metric.labels.response_code_class="5xx"
    comparison: COMPARISON_GT
    thresholdValue: 0.05
    duration: 600s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_RATE
alertStrategy:
  autoClose: 86400s
YAML
)"

# Cloud SQL CPU > 80% over 15 min.
write_policy "deepvogue-cloudsql-cpu" "$(cat <<'YAML'
displayName: deepvogue-cloudsql-cpu
combiner: OR
conditions:
- displayName: CPU > 80% (15m)
  conditionThreshold:
    filter: |
      metric.type="cloudsql.googleapis.com/database/cpu/utilization"
      resource.type="cloudsql_database"
    comparison: COMPARISON_GT
    thresholdValue: 0.8
    duration: 900s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
alertStrategy:
  autoClose: 86400s
YAML
)"

# Uptime-check failure (any of the deepvogue-* uptime checks).
write_policy "deepvogue-uptime-failure" "$(cat <<'YAML'
displayName: deepvogue-uptime-failure
combiner: OR
conditions:
- displayName: any deepvogue uptime check failing
  conditionThreshold:
    filter: |
      metric.type="monitoring.googleapis.com/uptime_check/check_passed"
      metric.labels.check_id=monitoring.regex.full_match("uptime-deepvogue-.*")
    comparison: COMPARISON_LT
    thresholdValue: 1
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_NEXT_OLDER
      crossSeriesReducer: REDUCE_COUNT_FALSE
      groupByFields:
      - resource.label.host
alertStrategy:
  autoClose: 86400s
YAML
)"

echo
echo "Monitoring setup done. Slack channel: ${SLACK_CHANNEL_DISPLAY}"
[ -z "${SLACK_WEBHOOK_URL:-}" ] && echo "(re-run with SLACK_WEBHOOK_URL set to attach notifications)"
