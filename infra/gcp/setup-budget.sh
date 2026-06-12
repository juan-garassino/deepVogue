#!/usr/bin/env bash
# Per-project budget + alerts on garassino-ml. Idempotent (upsert by displayName).
# Reads the project's billing account; skips gracefully if not linked.
#
# Required env: GCP_PROJECT.
# Optional env: DV_BUDGET_EUR (default 25), DV_BUDGET_EMAIL (default juan.garassino@hotmail.com).

# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

BUDGET_AMOUNT="${DV_BUDGET_EUR:-25}"
BUDGET_EMAIL="${DV_BUDGET_EMAIL:-juan.garassino@hotmail.com}"
BUDGET_NAME="deepvogue-${BUDGET_AMOUNT}-eur"

step "Resolve billing account for $PROJECT"
BILLING=$(gcloud billing projects describe "$PROJECT" \
  --format='value(billingAccountName)' 2>/dev/null || true)
if [ -z "$BILLING" ]; then
  echo "WARN: $PROJECT has no billing account linked — skipping budget setup"
  exit 0
fi
echo "billing: $BILLING"

# `gcloud beta billing budgets` is the maintained CLI; alpha is deprecated.
step "Upsert budget $BUDGET_NAME (€${BUDGET_AMOUNT}/mo, alerts 40/80/100%)"
EXISTING=$(gcloud beta billing budgets list --billing-account="${BILLING##*/}" \
  --filter="displayName=\"${BUDGET_NAME}\"" --format='value(name)' 2>/dev/null | head -n1 || true)

THRESHOLDS=(0.4 0.8 1.0)
THRESHOLD_FLAGS=()
for t in "${THRESHOLDS[@]}"; do
  THRESHOLD_FLAGS+=(--threshold-rule="percent=$t,basis=current-spend")
done

if [ -n "$EXISTING" ]; then
  gcloud beta billing budgets update "$EXISTING" \
    --display-name="$BUDGET_NAME" \
    --budget-amount="${BUDGET_AMOUNT}EUR" \
    --filter-projects="projects/$PROJECT" \
    "${THRESHOLD_FLAGS[@]}" \
    --quiet
  echo "updated budget: $BUDGET_NAME"
else
  gcloud beta billing budgets create \
    --billing-account="${BILLING##*/}" \
    --display-name="$BUDGET_NAME" \
    --budget-amount="${BUDGET_AMOUNT}EUR" \
    --filter-projects="projects/$PROJECT" \
    "${THRESHOLD_FLAGS[@]}"
  echo "created budget: $BUDGET_NAME"
fi

# Email notifications: the Cloud Billing API attaches them via the budget's
# notificationsRule, which the CLI doesn't fully cover. The simplest reliable
# path is a one-time portal step or a `gcloud alpha monitoring channels create
# --type=email` plus a notificationsRule patch — captured here as instructions
# so the operator wires it once.
echo
echo "Once-per-project step (not idempotent via CLI):"
echo "  console.cloud.google.com/billing/${BILLING##*/}/budgets → ${BUDGET_NAME} → Manage notifications →"
echo "  add email recipient: ${BUDGET_EMAIL}"
