#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${CI_RETRIEVAL_OUT_DIR:-docs_tmp/RAG/ci/local/${MODE}}"
mkdir -p "${OUT_DIR}"

CASES="tests/fixtures/retrieval_parity_cases.json"
CHUNK_RUNS="tests/fixtures/retrieval_parity_chunk_runs.json"
SEGMENT_RUNS="tests/fixtures/retrieval_parity_segment_runs.json"

case "${MODE}" in
  smoke)
    python scripts/retrieval_parity.py \
      --cases-json "$CASES" \
      --chunk-runs-json "$CHUNK_RUNS" \
      --segment-runs-json "$SEGMENT_RUNS" \
      --k 3 \
      --chunk-latency-ms 100 \
      --segment-latency-ms 110 \
      --min-overlap "${CI_RETRIEVAL_PARITY_SMOKE_MIN_OVERLAP:-0.60}" \
      --max-latency-ratio "${CI_RETRIEVAL_PARITY_SMOKE_MAX_LATENCY_RATIO:-1.50}" \
      --max-mrr-drop "${CI_RETRIEVAL_PARITY_SMOKE_MAX_MRR_DROP:-0.15}" \
      --output "${OUT_DIR}/retrieval_parity_smoke_${STAMP}.json"
    ;;
  nightly)
    python scripts/retrieval_parity.py \
      --cases-json "$CASES" \
      --chunk-runs-json "$CHUNK_RUNS" \
      --segment-runs-json "$SEGMENT_RUNS" \
      --k 3 \
      --chunk-latency-ms 100 \
      --segment-latency-ms 110 \
      --min-overlap "${CI_RETRIEVAL_PARITY_NIGHTLY_MIN_OVERLAP:-0.70}" \
      --max-latency-ratio "${CI_RETRIEVAL_PARITY_NIGHTLY_MAX_LATENCY_RATIO:-1.30}" \
      --max-mrr-drop "${CI_RETRIEVAL_PARITY_NIGHTLY_MAX_MRR_DROP:-0.10}" \
      --output "${OUT_DIR}/retrieval_parity_nightly_${STAMP}.json"
    ;;
  heavy)
    python scripts/retrieval_parity.py \
      --cases-json "$CASES" \
      --chunk-runs-json "$CHUNK_RUNS" \
      --segment-runs-json "$SEGMENT_RUNS" \
      --k 3 \
      --chunk-latency-ms 100 \
      --segment-latency-ms 110 \
      --min-overlap "${CI_RETRIEVAL_PARITY_HEAVY_MIN_OVERLAP:-0.75}" \
      --max-latency-ratio "${CI_RETRIEVAL_PARITY_HEAVY_MAX_LATENCY_RATIO:-1.20}" \
      --max-mrr-drop "${CI_RETRIEVAL_PARITY_HEAVY_MAX_MRR_DROP:-0.08}" \
      --output "${OUT_DIR}/retrieval_parity_heavy_${STAMP}.json"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "ci_retrieval_parity complete mode=${MODE} out_dir=${OUT_DIR}"
