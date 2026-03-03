#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-all}"

run_lint() {
  echo "[sdk-quality] lint (ruff)"
  uv run --project sdk/python --extra dev ruff check sdk/python/src sdk/python/tests
}

run_type() {
  echo "[sdk-quality] type (mypy)"
  uv run --project sdk/python --extra dev mypy sdk/python/src/querylake_sdk
}

run_test() {
  echo "[sdk-quality] test (pytest)"
  uv run --project sdk/python --extra dev pytest sdk/python/tests -q -m "not integration_live"
}

case "${MODE}" in
  lint)
    run_lint
    ;;
  type)
    run_type
    ;;
  test)
    run_test
    ;;
  all)
    run_lint
    run_type
    run_test
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [lint|type|test|all]" >&2
    exit 2
    ;;
esac
