#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

POINTER="${BCAS_BASELINE_POINTER:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_POINTER.json}"
DRILL_BASELINE="${BCAS_DRILL_BASELINE:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_2026-02-24.json}"
CANDIDATE="${BCAS_DRILL_CANDIDATE:-docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_phrase_candidate_rerank_2026-02-24.json}"
OUT_DIR="${BCAS_DRILL_OUT_DIR:-docs_tmp/RAG/drills}"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_POINTER="${OUT_DIR}/BCAS_PHASE2_OPERATOR_BASELINE_POINTER_BACKUP_${STAMP}.json"
ROLLBACK_OUT="${OUT_DIR}/BCAS_PHASE2_BASELINE_ROLLBACK_DRILL_${STAMP}.json"
RESTORE_OUT="${OUT_DIR}/BCAS_PHASE2_BASELINE_RESTORE_DRILL_${STAMP}.json"
GATE_ROLLBACK_OUT="${OUT_DIR}/BCAS_PHASE2_OPERATOR_GATE_ROLLBACK_DRILL_${STAMP}.json"
GATE_RESTORE_OUT="${OUT_DIR}/BCAS_PHASE2_OPERATOR_GATE_POST_RESTORE_DRILL_${STAMP}.json"
SUMMARY_OUT="${OUT_DIR}/BCAS_PHASE2_ROLLBACK_DRILL_SUMMARY_${STAMP}.json"

if [[ ! -f "${DRILL_BASELINE}" ]]; then
  echo "Drill baseline missing: ${DRILL_BASELINE}" >&2
  exit 2
fi
if [[ ! -f "${CANDIDATE}" ]]; then
  echo "Candidate artifact missing: ${CANDIDATE}" >&2
  exit 2
fi

if [[ -f "${POINTER}" ]]; then
  cp -f "${POINTER}" "${BACKUP_POINTER}"
fi

RESTORE_BASELINE="$(python - "${POINTER}" "${DRILL_BASELINE}" <<'PY'
import json
import sys
from pathlib import Path

pointer = Path(sys.argv[1])
fallback = sys.argv[2]
if pointer.exists():
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        value = payload.get("baseline_path")
        if isinstance(value, str) and len(value.strip()) > 0:
            print(value.strip())
            raise SystemExit(0)
    except Exception:
        pass
print(fallback)
PY
)"

python scripts/bcas_phase2_baseline_rollback.py \
  --pointer "${POINTER}" \
  --baseline-path "${DRILL_BASELINE}" \
  --reason "automated_rollback_drill" > "${ROLLBACK_OUT}"

python scripts/bcas_phase2_operator_gate.py \
  --baseline-pointer "${POINTER}" \
  --baseline-fallback "${DRILL_BASELINE}" \
  --candidate "${CANDIDATE}" \
  --out "${GATE_ROLLBACK_OUT}" || true

python scripts/bcas_phase2_baseline_rollback.py \
  --pointer "${POINTER}" \
  --baseline-path "${RESTORE_BASELINE}" \
  --reason "automated_restore_after_drill" > "${RESTORE_OUT}"

python scripts/bcas_phase2_operator_gate.py \
  --baseline-pointer "${POINTER}" \
  --baseline-fallback "${RESTORE_BASELINE}" \
  --candidate "${CANDIDATE}" \
  --out "${GATE_RESTORE_OUT}" || true

python - <<'PY' "${SUMMARY_OUT}" "${POINTER}" "${BACKUP_POINTER}" "${ROLLBACK_OUT}" "${RESTORE_OUT}" "${GATE_ROLLBACK_OUT}" "${GATE_RESTORE_OUT}" "${DRILL_BASELINE}" "${RESTORE_BASELINE}"
import json
import time
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
def _load(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

gate_rollback = _load(sys.argv[6])
gate_restore = _load(sys.argv[7])
payload = {
    "generated_at_unix": time.time(),
    "pointer": sys.argv[2],
    "pointer_backup": sys.argv[3],
    "rollback_artifact": sys.argv[4],
    "restore_artifact": sys.argv[5],
    "gate_after_rollback": sys.argv[6],
    "gate_after_restore": sys.argv[7],
    "drill_baseline": sys.argv[8],
    "restore_baseline": sys.argv[9],
    "gate_after_rollback_ok": bool(gate_rollback.get("gate_ok", False)),
    "gate_after_restore_ok": bool(gate_restore.get("gate_ok", False)),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "rollback_drill complete summary=${SUMMARY_OUT}"
