#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[ci-sdk] cleaning prior build artifacts"
rm -rf sdk/python/dist

echo "[ci-sdk] running SDK quality gate (lint + type + tests)"
bash scripts/dev/sdk_quality_gate.sh all

echo "[ci-sdk] building SDK package"
uv run --project sdk/python --with build python -m build sdk/python

echo "[ci-sdk] validating package metadata"
uv run --project sdk/python --with twine python -m twine check sdk/python/dist/*

echo "[ci-sdk] verifying wheel contents"
uv run --no-project python scripts/dev/verify_sdk_wheel.py --dist-dir sdk/python/dist

echo "[ci-sdk] done"
