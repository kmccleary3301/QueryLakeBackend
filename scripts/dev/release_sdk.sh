#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

REPO_TARGET="${1:-check}"
# Allowed:
#   check     -> test + build + twine check only
#   testpypi  -> upload to TestPyPI
#   pypi      -> upload to PyPI

SDK_PACKAGE_NAME="querylake-sdk"
GITHUB_REF_VALUE="${GITHUB_REF:-}"
SKIP_REMOTE_GUARD="${SDK_RELEASE_SKIP_REMOTE_CHECK:-0}"

echo "[sdk-release] target=${REPO_TARGET}"
echo "[sdk-release] running SDK CI checks (tests + build + metadata + wheel validation)"
bash scripts/ci_sdk_checks.sh

if [[ "${REPO_TARGET}" == "check" ]]; then
  echo "[sdk-release] done (check-only mode)"
  exit 0
fi

if [[ "${REPO_TARGET}" == "testpypi" ]]; then
  echo "[sdk-release] validating publish guard (target=testpypi package=${SDK_PACKAGE_NAME})"
  GUARD_ARGS=(--target testpypi --package-name "${SDK_PACKAGE_NAME}")
  if [[ -n "${GITHUB_REF_VALUE}" ]]; then
    GUARD_ARGS+=(--github-ref "${GITHUB_REF_VALUE}")
  fi
  if [[ "${SKIP_REMOTE_GUARD}" == "1" ]]; then
    GUARD_ARGS+=(--skip-remote-check)
  fi
  uv run --no-project python scripts/dev/verify_sdk_publish_guard.py "${GUARD_ARGS[@]}"
  echo "[sdk-release] uploading to TestPyPI"
  uv run --project sdk/python --with twine python -m twine upload --repository testpypi sdk/python/dist/*
  exit 0
fi

if [[ "${REPO_TARGET}" == "pypi" ]]; then
  echo "[sdk-release] validating publish guard (target=pypi package=${SDK_PACKAGE_NAME})"
  GUARD_ARGS=(--target pypi --package-name "${SDK_PACKAGE_NAME}")
  if [[ -n "${GITHUB_REF_VALUE}" ]]; then
    GUARD_ARGS+=(--github-ref "${GITHUB_REF_VALUE}")
  fi
  if [[ "${SKIP_REMOTE_GUARD}" == "1" ]]; then
    GUARD_ARGS+=(--skip-remote-check)
  fi
  uv run --no-project python scripts/dev/verify_sdk_publish_guard.py "${GUARD_ARGS[@]}"
  echo "[sdk-release] uploading to PyPI"
  uv run --project sdk/python --with twine python -m twine upload sdk/python/dist/*
  exit 0
fi

echo "[sdk-release] unknown target: ${REPO_TARGET}"
echo "usage: ./scripts/dev/release_sdk.sh [check|testpypi|pypi]"
exit 2
