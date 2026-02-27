#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DAY_STAMP="$(date +%Y-%m-%d)"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"

OUT_DIR="${BCAS_NIGHTLY_STRICT_OUT_DIR:-docs_tmp/RAG/nightly_strict/${DAY_STAMP}}"
mkdir -p "${OUT_DIR}"

export CI_RETRIEVAL_OUT_DIR="${CI_RETRIEVAL_OUT_DIR:-${OUT_DIR}}"
export BCAS_STRICT_PRESET="${BCAS_STRICT_PRESET:-strict_embedreuse_v1}"
export BCAS_STRICT_TRACK_OUT="${BCAS_STRICT_TRACK_OUT:-${OUT_DIR}/BCAS_PHASE2_3LANE_TRACK_nightly_strict_${RUN_STAMP}.json}"
export BCAS_STRICT_RELEASE_GATE_JSON="${BCAS_STRICT_RELEASE_GATE_JSON:-${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_nightly_strict_${RUN_STAMP}.json}"
export BCAS_STRICT_RELEASE_GATE_MD="${BCAS_STRICT_RELEASE_GATE_MD:-${OUT_DIR}/BCAS_PHASE2_RELEASE_GATE_nightly_strict_${RUN_STAMP}.md}"
export BCAS_STRICT_NOTIFY_OUT="${BCAS_STRICT_NOTIFY_OUT:-${OUT_DIR}/BCAS_PHASE2_NOTIFICATIONS_nightly_strict_${RUN_STAMP}.json}"

echo "[nightly-strict] out_dir=${OUT_DIR}"
echo "[nightly-strict] preset=${BCAS_STRICT_PRESET}"

bash scripts/bcas_phase2_strict_check.sh

echo "[nightly-strict] complete"
echo "[nightly-strict] track=${BCAS_STRICT_TRACK_OUT}"
echo "[nightly-strict] release_gate_json=${BCAS_STRICT_RELEASE_GATE_JSON}"
echo "[nightly-strict] release_gate_md=${BCAS_STRICT_RELEASE_GATE_MD}"
echo "[nightly-strict] notify=${BCAS_STRICT_NOTIFY_OUT}"
