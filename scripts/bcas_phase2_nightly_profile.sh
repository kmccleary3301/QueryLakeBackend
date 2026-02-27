#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y-%m-%d)"
OUT_DIR="docs_tmp/RAG/nightly/${STAMP}"
mkdir -p "$OUT_DIR"

ACCOUNT_CONFIG="${ACCOUNT_CONFIG:-docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json}"
SAMPLES_ROOT="${SAMPLES_ROOT:-docs_tmp/RAG/ingest_logs_2026-02-24_aug}"
BASELINE_POINTER="${BASELINE_POINTER:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_POINTER.json}"
BASELINE_FALLBACK="${BASELINE_FALLBACK:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_2026-02-24.json}"
BASELINE_PROPOSAL_OUT="${BASELINE_PROPOSAL_OUT:-$OUT_DIR/BCAS_PHASE2_OPERATOR_BASELINE_PROPOSAL_${STAMP}.json}"
POLICY_LATEST="${POLICY_LATEST:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_LATEST.json}"
WEBHOOK_SECRET_FILE="${BCAS_PHASE2_NOTIFY_WEBHOOK_SECRET_FILE:-}"

if [[ -z "${BCAS_PHASE2_NOTIFY_WEBHOOK:-}" ]]; then
  if [[ -n "$WEBHOOK_SECRET_FILE" && -f "$WEBHOOK_SECRET_FILE" ]]; then
    export BCAS_PHASE2_NOTIFY_WEBHOOK="$(tr -d '\r\n' < "$WEBHOOK_SECRET_FILE")"
  elif [[ -f "/run/secrets/bcas_phase2_notify_webhook" ]]; then
    export BCAS_PHASE2_NOTIFY_WEBHOOK="$(tr -d '\r\n' < "/run/secrets/bcas_phase2_notify_webhook")"
  fi
fi

echo "[nightly] output directory: $OUT_DIR"

python scripts/bcas_phase2_eval.py \
  --account-config "$ACCOUNT_CONFIG" \
  --samples-root "$SAMPLES_ROOT" \
  --per-dataset-cases 80 \
  --bm25-weight 0.4 \
  --similarity-weight 0.6 \
  --limit-bm25 20 \
  --limit-similarity 20 \
  --cases-out "$OUT_DIR/BCAS_PHASE2_LIVE_EVAL_CASES_dense_heavy_augtrivia_${STAMP}.json" \
  --metrics-out "$OUT_DIR/BCAS_PHASE2_LIVE_EVAL_METRICS_dense_heavy_augtrivia_${STAMP}.json"

python scripts/retrieval_eval.py \
  --mode smoke \
  --input "$OUT_DIR/BCAS_PHASE2_LIVE_EVAL_CASES_dense_heavy_augtrivia_${STAMP}.json" \
  --output "$OUT_DIR/BCAS_PHASE2_LIVE_EVAL_RETRIEVAL_EVAL_dense_heavy_augtrivia_${STAMP}.json" \
  --k 10 \
  --min-recall 0 \
  --min-mrr 0

python scripts/bcas_phase2_operator_eval.py \
  --account-config "$ACCOUNT_CONFIG" \
  --samples-root "$SAMPLES_ROOT" \
  --out "$OUT_DIR/BCAS_PHASE2_OPERATOR_EVAL_${STAMP}.json"

python scripts/bcas_phase2_stress.py \
  --account-config "$ACCOUNT_CONFIG" \
  --cases "$OUT_DIR/BCAS_PHASE2_LIVE_EVAL_CASES_dense_heavy_augtrivia_${STAMP}.json" \
  --duration-s 45 \
  --concurrency 8 \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --out "$OUT_DIR/BCAS_PHASE2_STRESS_c8_${STAMP}.json"

python scripts/bcas_phase2_operator_gate.py \
  --baseline-pointer "$BASELINE_POINTER" \
  --baseline-fallback "$BASELINE_FALLBACK" \
  --policy "$POLICY_LATEST" \
  --use-policy-if-ready \
  --enforce-profile-match \
  --candidate "$OUT_DIR/BCAS_PHASE2_OPERATOR_EVAL_${STAMP}.json" \
  --out "$OUT_DIR/BCAS_PHASE2_OPERATOR_GATE_${STAMP}.json"

python scripts/bcas_phase2_baseline_rollover.py \
  --action propose \
  --pointer "$BASELINE_POINTER" \
  --proposal "$BASELINE_PROPOSAL_OUT" \
  --nightly-root "docs_tmp/RAG/nightly" \
  --min-consecutive-days "${ROLLOVER_MIN_CONSECUTIVE_DAYS:-3}" \
  --min-exact-pass-delta "${ROLLOVER_MIN_EXACT_PASS_DELTA:-0.0}" \
  --max-exact-latency-delta-ms "${ROLLOVER_MAX_EXACT_LATENCY_DELTA_MS:-5.0}" \
  --max-days-scan "${ROLLOVER_MAX_DAYS_SCAN:-14}"

python scripts/bcas_phase2_notify.py \
  --gate "$OUT_DIR/BCAS_PHASE2_OPERATOR_GATE_${STAMP}.json" \
  --stress "$OUT_DIR/BCAS_PHASE2_STRESS_c8_${STAMP}.json" \
  --out "$OUT_DIR/BCAS_PHASE2_NOTIFICATIONS_${STAMP}.json" \
  --webhook-secret-file "${WEBHOOK_SECRET_FILE:-/run/secrets/bcas_phase2_notify_webhook}" \
  --state-file "docs_tmp/RAG/BCAS_PHASE2_NOTIFY_STATE.json" \
  --cooldown-seconds "${ALERT_COOLDOWN_SECONDS:-900}" \
  --p95-threshold-ms "${ALERT_P95_THRESHOLD_MS:-4500}" \
  --p99-threshold-ms "${ALERT_P99_THRESHOLD_MS:-5500}"

python scripts/bcas_phase2_nightly_trend.py \
  --nightly-root "docs_tmp/RAG/nightly" \
  --max-days "${TREND_MAX_DAYS:-7}" \
  --out "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_${STAMP}.json"

python scripts/bcas_phase2_daily_status.py \
  --trend-json "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_${STAMP}.json" \
  --out "docs_tmp/RAG/BCAS_PHASE2_DAILY_STATUS_${STAMP}.md"

python scripts/bcas_phase2_threshold_policy.py \
  --nightly-root "docs_tmp/RAG/nightly" \
  --max-days "${POLICY_MAX_DAYS:-14}" \
  --min-days "${POLICY_MIN_DAYS:-5}" \
  --out "docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_${STAMP}.json" \
  --latest-out "$POLICY_LATEST"

python scripts/bcas_phase2_oncall_dashboard.py \
  --nightly-root "docs_tmp/RAG/nightly" \
  --trend "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_${STAMP}.json" \
  --policy "docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_${STAMP}.json" \
  --out "docs_tmp/RAG/BCAS_PHASE2_ONCALL_DASHBOARD_${STAMP}.md"

python scripts/bcas_phase2_soak_tracker.py \
  --nightly-root "docs_tmp/RAG/nightly" \
  --required-days "${SOAK_REQUIRED_DAYS:-7}" \
  --max-days "${SOAK_MAX_DAYS:-21}" \
  --out "docs_tmp/RAG/BCAS_PHASE2_SOAK_TRACKER_${STAMP}.json"

python scripts/bcas_phase2_trend_review.py \
  --trend "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_${STAMP}.json" \
  --soak "docs_tmp/RAG/BCAS_PHASE2_SOAK_TRACKER_${STAMP}.json" \
  --out-json "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_${STAMP}.json" \
  --out-md "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_${STAMP}.md" \
  --min-avg-recall "${TREND_MIN_AVG_RECALL:-0.15}" \
  --min-avg-mrr "${TREND_MIN_AVG_MRR:-0.15}" \
  --min-avg-operator-pass "${TREND_MIN_AVG_OPERATOR_PASS:-0.90}" \
  --max-avg-stress-p95-ms "${TREND_MAX_AVG_STRESS_P95_MS:-4500}"

cp -f "docs_tmp/RAG/BCAS_PHASE2_ONCALL_DASHBOARD_${STAMP}.md" "docs_tmp/RAG/BCAS_PHASE2_ONCALL_DASHBOARD_LATEST.md"
cp -f "docs_tmp/RAG/BCAS_PHASE2_DAILY_STATUS_${STAMP}.md" "docs_tmp/RAG/BCAS_PHASE2_DAILY_STATUS_LATEST.md"
cp -f "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_${STAMP}.md" "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_LATEST.md"
cp -f "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_${STAMP}.json" "docs_tmp/RAG/BCAS_PHASE2_TREND_REVIEW_LATEST.json"

if python scripts/bcas_phase2_nightly_delta.py \
  --nightly-root "docs_tmp/RAG/nightly" \
  --out-json "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_${STAMP}.json" \
  --out-md "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_${STAMP}.md"; then
  cp -f "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_${STAMP}.json" "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_LATEST.json"
  cp -f "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_${STAMP}.md" "docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_DELTA_LATEST.md"
else
  echo "[nightly] delta report skipped (need >=2 nightly day dirs and matching eval/stress artifacts)"
fi

echo "[nightly] complete"
