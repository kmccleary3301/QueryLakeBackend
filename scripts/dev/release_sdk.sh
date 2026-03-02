#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

REPO_TARGET="${1:-check}"
# Allowed:
#   check     -> test + build + twine check only
#   testpypi  -> upload to TestPyPI
#   pypi      -> upload to PyPI

echo "[sdk-release] target=${REPO_TARGET}"
echo "[sdk-release] running SDK CI checks (tests + build + metadata + wheel validation)"
bash scripts/ci_sdk_checks.sh

if [[ "${REPO_TARGET}" == "check" ]]; then
  echo "[sdk-release] done (check-only mode)"
  exit 0
fi

if [[ "${REPO_TARGET}" == "testpypi" ]]; then
  echo "[sdk-release] uploading to TestPyPI"
  uv run --project sdk/python --with twine python -m twine upload --repository testpypi sdk/python/dist/*
  exit 0
fi

if [[ "${REPO_TARGET}" == "pypi" ]]; then
  echo "[sdk-release] uploading to PyPI"
  uv run --project sdk/python --with twine python -m twine upload sdk/python/dist/*
  exit 0
fi

echo "[sdk-release] unknown target: ${REPO_TARGET}"
echo "usage: ./scripts/dev/release_sdk.sh [check|testpypi|pypi]"
exit 2
