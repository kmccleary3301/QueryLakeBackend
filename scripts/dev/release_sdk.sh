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
echo "[sdk-release] running SDK unit tests"
uv run --project sdk/python pytest sdk/python/tests -q

echo "[sdk-release] building artifacts"
uv run --project sdk/python --with build python -m build sdk/python

echo "[sdk-release] validating package metadata"
uv run --project sdk/python --with twine python -m twine check sdk/python/dist/*

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
