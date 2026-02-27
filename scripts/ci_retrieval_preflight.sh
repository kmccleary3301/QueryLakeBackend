#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
OUT_DIR="${CI_RETRIEVAL_OUT_DIR:-docs_tmp/RAG/ci/local/${MODE}}"
mkdir -p "${OUT_DIR}"
OUT_JSON="${CI_RETRIEVAL_PREFLIGHT_OUT:-${OUT_DIR}/preflight_${MODE}.json}"

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    return 1
  fi
  return 0
}

check_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    return 1
  fi
  return 0
}

check_cmd python
check_file scripts/retrieval_eval.py
check_file scripts/retrieval_parity.py
check_file tests/fixtures/retrieval_eval_smoke.json
check_file tests/fixtures/retrieval_parity_cases.json
check_file tests/fixtures/retrieval_parity_chunk_runs.json
check_file tests/fixtures/retrieval_parity_segment_runs.json

heavy_account_config="${BCAS_ACCOUNT_CONFIG:-docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json}"
heavy_api_base="${BCAS_API_BASE_URL:-http://localhost:8000}"
heavy_enabled=false

if [[ "${MODE}" == "nightly" || "${MODE}" == "heavy" ]]; then
  check_file tests/fixtures/retrieval_eval_nightly.json
fi

if [[ "${MODE}" == "heavy" ]]; then
  heavy_enabled=true
  check_file scripts/bcas_phase2_strict_check.sh
  check_file scripts/bcas_phase2_release_gate.py
  if [[ "${CI_RETRIEVAL_PREFLIGHT_ALLOW_MISSING_BCAS_CONFIG:-0}" != "1" ]]; then
    check_file "${heavy_account_config}"
  fi
fi

python - <<'PY' "${OUT_JSON}" "${MODE}" "${heavy_enabled}" "${heavy_account_config}" "${heavy_api_base}"
import json
import time
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
mode = sys.argv[2]
heavy_enabled = (sys.argv[3].lower() == "true")
heavy_account_config = sys.argv[4]
heavy_api_base = sys.argv[5]

payload = {
    "mode": mode,
    "ok": True,
    "generated_at_unix": time.time(),
    "checks": {
        "python": True,
        "eval_script": True,
        "parity_script": True,
        "fixtures": True,
        "heavy_enabled": heavy_enabled,
        "heavy_account_config": heavy_account_config if heavy_enabled else None,
        "heavy_api_base_url": heavy_api_base if heavy_enabled else None,
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Preflight passed for mode=${MODE}"
