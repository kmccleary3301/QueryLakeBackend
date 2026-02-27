#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${CI_RETRIEVAL_OUT_DIR:-docs_tmp/RAG/ci/local/${MODE}}"
mkdir -p "${OUT_DIR}"

case "${MODE}" in
  smoke)
    python scripts/retrieval_eval.py \
      --mode smoke \
      --min-recall "${CI_RETRIEVAL_SMOKE_MIN_RECALL:-0.50}" \
      --min-mrr "${CI_RETRIEVAL_SMOKE_MIN_MRR:-0.40}" \
      --output "${OUT_DIR}/retrieval_eval_smoke_${STAMP}.json"
    ;;
  nightly)
    python scripts/retrieval_eval.py \
      --mode nightly \
      --min-recall "${CI_RETRIEVAL_NIGHTLY_MIN_RECALL:-0.50}" \
      --min-mrr "${CI_RETRIEVAL_NIGHTLY_MIN_MRR:-0.40}" \
      --output "${OUT_DIR}/retrieval_eval_nightly_${STAMP}.json"
    ;;
  heavy)
    python scripts/retrieval_eval.py \
      --mode heavy \
      --input "${CI_RETRIEVAL_HEAVY_INPUT:-tests/fixtures/retrieval_eval_agentic_depth.json}" \
      --min-recall "${CI_RETRIEVAL_HEAVY_MIN_RECALL:-0.55}" \
      --min-mrr "${CI_RETRIEVAL_HEAVY_MIN_MRR:-0.45}" \
      --output "${OUT_DIR}/retrieval_eval_heavy_${STAMP}.json"

    if [[ "${CI_RETRIEVAL_HEAVY_ENABLE_BCAS:-1}" == "1" ]]; then
      BCAS_STRICT_TRACK_OUT="${OUT_DIR}/BCAS_PHASE2_3LANE_TRACK_heavy_${STAMP}.json" \
      BCAS_STRICT_RELEASE_GATE_JSON="${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_heavy_${STAMP}.json" \
      BCAS_STRICT_RELEASE_GATE_MD="${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_heavy_${STAMP}.md" \
      bash scripts/bcas_phase2_strict_check.sh
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "ci_retrieval_eval complete mode=${MODE} out_dir=${OUT_DIR}"
