#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${CI_RETRIEVAL_OUT_DIR:-docs_tmp/RAG/ci/local/heavy}"
mkdir -p "${OUT_DIR}"

TRACK_OUT="${BCAS_STRICT_TRACK_OUT:-${OUT_DIR}/BCAS_PHASE2_3LANE_TRACK_strict_${STAMP}.json}"
GATE_OUT_JSON="${BCAS_STRICT_RELEASE_GATE_JSON:-${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_strict_${STAMP}.json}"
GATE_OUT_MD="${BCAS_STRICT_RELEASE_GATE_MD:-${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_strict_${STAMP}.md}"
POLICY_PATH="${BCAS_STRICT_POLICY_PATH:-docs_tmp/RAG/BCAS_PHASE2_STRICT_PRESET_POLICY_2026-02-25.json}"
STRICT_PRESET="${BCAS_STRICT_PRESET:-strict_embedreuse_v1}"

# Named strict presets.
case "${STRICT_PRESET}" in
  strict_embedreuse_v1)
    : "${BCAS_STRICT_LIMIT_BM25:=15}"
    : "${BCAS_STRICT_LIMIT_SIMILARITY:=15}"
    : "${BCAS_STRICT_CANDIDATE_LIMIT_SPARSE:=8}"
    : "${BCAS_STRICT_CANDIDATE_SPARSE_WEIGHT:=0.12}"
    : "${BCAS_STRICT_QUEUE_SOFT_UTILIZATION:=0.75}"
    : "${BCAS_STRICT_QUEUE_HARD_UTILIZATION:=0.90}"
    : "${BCAS_STRICT_QUEUE_SOFT_SCALE:=0.60}"
    : "${BCAS_STRICT_QUEUE_HARD_SCALE:=0.35}"
    : "${BCAS_STRICT_QUEUE_DISABLE_SPARSE_AT_HARD:=1}"
    ;;
  strict_queue_confirm_qbase12)
    : "${BCAS_STRICT_LIMIT_BM25:=15}"
    : "${BCAS_STRICT_LIMIT_SIMILARITY:=15}"
    : "${BCAS_STRICT_CANDIDATE_LIMIT_SPARSE:=10}"
    : "${BCAS_STRICT_CANDIDATE_SPARSE_WEIGHT:=0.15}"
    : "${BCAS_STRICT_QUEUE_SOFT_UTILIZATION:=0.75}"
    : "${BCAS_STRICT_QUEUE_HARD_UTILIZATION:=0.90}"
    : "${BCAS_STRICT_QUEUE_SOFT_SCALE:=0.75}"
    : "${BCAS_STRICT_QUEUE_HARD_SCALE:=0.50}"
    : "${BCAS_STRICT_QUEUE_DISABLE_SPARSE_AT_HARD:=1}"
    ;;
  *)
    echo "Unknown BCAS_STRICT_PRESET='${STRICT_PRESET}' (expected strict_embedreuse_v1 or strict_queue_confirm_qbase12)" >&2
    exit 2
    ;;
esac

queue_disable_sparse_flag="--no-queue-throttle-disable-sparse-at-hard"
case "$(printf '%s' "${BCAS_STRICT_QUEUE_DISABLE_SPARSE_AT_HARD}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    queue_disable_sparse_flag="--queue-throttle-disable-sparse-at-hard"
    ;;
esac

echo "strict_check preset=${STRICT_PRESET}"
echo "strict_check profile limit_bm25=${BCAS_STRICT_LIMIT_BM25} limit_similarity=${BCAS_STRICT_LIMIT_SIMILARITY} candidate_limit_sparse=${BCAS_STRICT_CANDIDATE_LIMIT_SPARSE} candidate_sparse_weight=${BCAS_STRICT_CANDIDATE_SPARSE_WEIGHT}"
echo "strict_check queue soft_util=${BCAS_STRICT_QUEUE_SOFT_UTILIZATION} hard_util=${BCAS_STRICT_QUEUE_HARD_UTILIZATION} soft_scale=${BCAS_STRICT_QUEUE_SOFT_SCALE} hard_scale=${BCAS_STRICT_QUEUE_HARD_SCALE} disable_sparse_at_hard=${BCAS_STRICT_QUEUE_DISABLE_SPARSE_AT_HARD}"

python scripts/bcas_phase2_three_lane_track.py \
  --account-config "${BCAS_ACCOUNT_CONFIG:-docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json}" \
  --out-dir "${OUT_DIR}" \
  --out "${TRACK_OUT}" \
  --per-dataset-cases "${CI_RETRIEVAL_HEAVY_CASES:-80}" \
  --limit-bm25 "${BCAS_STRICT_LIMIT_BM25}" \
  --limit-similarity "${BCAS_STRICT_LIMIT_SIMILARITY}" \
  --candidate-limit-sparse "${BCAS_STRICT_CANDIDATE_LIMIT_SPARSE}" \
  --candidate-sparse-weight "${BCAS_STRICT_CANDIDATE_SPARSE_WEIGHT}" \
  --stress-duration-s "${CI_RETRIEVAL_HEAVY_STRESS_DURATION_S:-90}" \
  --stress-concurrency "${CI_RETRIEVAL_HEAVY_STRESS_CONCURRENCY:-12}" \
  --queue-admission-concurrency-limit "${CI_RETRIEVAL_HEAVY_QUEUE_LIMIT:-12}" \
  --queue-throttle-soft-utilization "${BCAS_STRICT_QUEUE_SOFT_UTILIZATION}" \
  --queue-throttle-hard-utilization "${BCAS_STRICT_QUEUE_HARD_UTILIZATION}" \
  --queue-throttle-soft-scale "${BCAS_STRICT_QUEUE_SOFT_SCALE}" \
  --queue-throttle-hard-scale "${BCAS_STRICT_QUEUE_HARD_SCALE}" \
  "${queue_disable_sparse_flag}"

readarray -t PATH_ROWS < <(python - "${TRACK_OUT}" <<'PY'
import json
import sys
from pathlib import Path

track_path = Path(sys.argv[1])
payload = json.loads(track_path.read_text(encoding="utf-8"))
artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
keys = [
    "baseline_eval_metrics",
    "candidate_eval_metrics",
    "baseline_stress",
    "candidate_stress",
]
for key in keys:
    print(artifacts.get(key, ""))
PY
)

BASELINE_EVAL="${PATH_ROWS[0]:-}"
CANDIDATE_EVAL="${PATH_ROWS[1]:-}"
BASELINE_STRESS="${PATH_ROWS[2]:-}"
CANDIDATE_STRESS="${PATH_ROWS[3]:-}"

if [[ -z "${BASELINE_EVAL}" || -z "${CANDIDATE_EVAL}" || -z "${BASELINE_STRESS}" || -z "${CANDIDATE_STRESS}" ]]; then
  echo "Could not resolve required artifact paths from ${TRACK_OUT}" >&2
  exit 2
fi

python scripts/bcas_phase2_release_gate.py \
  --baseline-eval "${BASELINE_EVAL}" \
  --candidate-eval "${CANDIDATE_EVAL}" \
  --baseline-stress "${BASELINE_STRESS}" \
  --candidate-stress "${CANDIDATE_STRESS}" \
  --policy "${POLICY_PATH}" \
  --out-json "${GATE_OUT_JSON}" \
  --out-md "${GATE_OUT_MD}" \
  --fail-on-gate

dashboard_root="${BCAS_STRICT_DASHBOARD_CI_ROOT:-}"
if [[ -z "${dashboard_root}" ]]; then
  if [[ "${OUT_DIR}" == *"/ci/"* || "${OUT_DIR}" == docs_tmp/RAG/ci/* ]]; then
    dashboard_root="docs_tmp/RAG/ci"
  else
    dashboard_root="$(dirname "$(dirname "${OUT_DIR}")")"
  fi
fi

python scripts/bcas_phase2_publish_strict_dashboard.py \
  --ci-root "${dashboard_root}" \
  --out "docs_tmp/RAG/BCAS_PHASE2_STRICT_DASHBOARD_LATEST.md"

if [[ "${BCAS_STRICT_NOTIFY_ENABLE:-1}" == "1" ]]; then
  notify_out="${BCAS_STRICT_NOTIFY_OUT:-${OUT_DIR}/BCAS_PHASE2_NOTIFICATIONS_strict_${STAMP}.json}"
  python scripts/bcas_phase2_notify.py \
    --gate "${GATE_OUT_JSON}" \
    --stress "${CANDIDATE_STRESS}" \
    --out "${notify_out}" \
    --state-file "${BCAS_STRICT_NOTIFY_STATE_FILE:-docs_tmp/RAG/BCAS_PHASE2_NOTIFY_STATE.json}" \
    --webhook-secret-file "${BCAS_PHASE2_NOTIFY_WEBHOOK_SECRET_FILE:-/run/secrets/bcas_phase2_notify_webhook}" \
    --cooldown-seconds "${ALERT_COOLDOWN_SECONDS:-900}" \
    --p95-threshold-ms "${ALERT_P95_THRESHOLD_MS:-4500}" \
    --p99-threshold-ms "${ALERT_P99_THRESHOLD_MS:-5500}" || true
fi

echo "strict_check complete"
echo "track=${TRACK_OUT}"
echo "release_gate_json=${GATE_OUT_JSON}"
echo "release_gate_md=${GATE_OUT_MD}"
