#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${BCAS_SOAK_OUT_DIR:-docs_tmp/RAG/soak/$(date +%Y-%m-%d)}"
mkdir -p "${OUT_DIR}"

DURATION_HOURS="${BCAS_SOAK_DURATION_HOURS:-12}"
INTERVAL_MINUTES="${BCAS_SOAK_INTERVAL_MINUTES:-30}"
MAX_RUNS="${BCAS_SOAK_MAX_RUNS:-0}" # 0 => derived from duration/interval

if [[ "${MAX_RUNS}" == "0" ]]; then
  MAX_RUNS=$(( (DURATION_HOURS * 60) / INTERVAL_MINUTES ))
fi
if [[ "${MAX_RUNS}" -le 0 ]]; then
  echo "Invalid run count computed from duration/interval." >&2
  exit 2
fi

SUMMARY_JSON="${BCAS_SOAK_SUMMARY_JSON:-${OUT_DIR}/BCAS_PHASE2_SOAK_SUMMARY_$(date +%Y%m%d-%H%M%S).json}"
RESULTS_JSONL="${BCAS_SOAK_RESULTS_JSONL:-${OUT_DIR}/BCAS_PHASE2_SOAK_RESULTS.jsonl}"

echo "Starting soak runs=${MAX_RUNS} interval_min=${INTERVAL_MINUTES} out_dir=${OUT_DIR}"

ok_runs=0
failed_runs=0
first_failure_index=-1

for ((i=1; i<=MAX_RUNS; i++)); do
  stamp="$(date +%Y%m%d-%H%M%S)"
  run_dir="${OUT_DIR}/run_${i}_${stamp}"
  mkdir -p "${run_dir}"

  rc=0
  CI_RETRIEVAL_OUT_DIR="${run_dir}" \
  CI_RETRIEVAL_HEAVY_CASES="${CI_RETRIEVAL_HEAVY_CASES:-80}" \
  CI_RETRIEVAL_HEAVY_STRESS_DURATION_S="${CI_RETRIEVAL_HEAVY_STRESS_DURATION_S:-90}" \
  CI_RETRIEVAL_HEAVY_STRESS_CONCURRENCY="${CI_RETRIEVAL_HEAVY_STRESS_CONCURRENCY:-12}" \
  CI_RETRIEVAL_HEAVY_QUEUE_LIMIT="${CI_RETRIEVAL_HEAVY_QUEUE_LIMIT:-12}" \
  bash scripts/bcas_phase2_strict_check.sh || rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    ok_runs=$((ok_runs + 1))
    run_status="ok"
  else
    failed_runs=$((failed_runs + 1))
    run_status="failed"
    if [[ "${first_failure_index}" -lt 0 ]]; then
      first_failure_index="${i}"
    fi
  fi

  python - <<'PY' "${RESULTS_JSONL}" "${i}" "${run_status}" "${rc}" "${run_dir}"
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "run_index": int(sys.argv[2]),
    "status": sys.argv[3],
    "return_code": int(sys.argv[4]),
    "run_dir": sys.argv[5],
    "timestamp_unix": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
PY

  if [[ "${i}" -lt "${MAX_RUNS}" ]]; then
    sleep "$((INTERVAL_MINUTES * 60))"
  fi
done

python - <<'PY' "${SUMMARY_JSON}" "${MAX_RUNS}" "${ok_runs}" "${failed_runs}" "${first_failure_index}" "${RESULTS_JSONL}" "${OUT_DIR}"
import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "generated_at_unix": time.time(),
    "planned_runs": int(sys.argv[2]),
    "ok_runs": int(sys.argv[3]),
    "failed_runs": int(sys.argv[4]),
    "first_failure_index": int(sys.argv[5]),
    "results_jsonl": sys.argv[6],
    "out_dir": sys.argv[7],
    "gate_ok": int(sys.argv[4]) == 0,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Soak complete summary=${SUMMARY_JSON}"
