#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[bootstrap] Syncing backend dependencies (dev + cli extras)"
uv sync --extra dev --extra cli

if [[ ! -f ".env" ]]; then
  echo "[bootstrap] Creating .env from .env.example"
  cp .env.example .env
fi

echo "[bootstrap] Installing SDK in editable mode"
uv run --project sdk/python pip install -e sdk/python

echo "[bootstrap] Done"
echo "Next:"
echo "  make up-db"
echo "  make run-api-only"
echo "  make health"
