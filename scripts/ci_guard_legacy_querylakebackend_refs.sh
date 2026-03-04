#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Historical migration documentation is allowed to mention the previous repo name.
ALLOW_EXCLUDES=(
  ":(exclude)docs/unification/repo_migration.md"
  ":(exclude)scripts/ci_guard_legacy_querylakebackend_refs.sh"
)

PATTERN='QueryLakeBackend|/shared_folders/querylake_server/QueryLakeBackend'

if MATCHES="$(git grep -n -I -E "$PATTERN" -- . "${ALLOW_EXCLUDES[@]}" || true)"; then
  :
fi

if [[ -n "${MATCHES// }" ]]; then
  echo "ERROR: Legacy QueryLakeBackend reference(s) detected."
  echo "Please migrate to the canonical QueryLake naming."
  echo
  echo "$MATCHES"
  exit 1
fi

echo "Legacy path/name guard passed."

