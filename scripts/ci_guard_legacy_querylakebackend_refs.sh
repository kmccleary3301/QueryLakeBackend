#!/usr/bin/env bash
set -euo pipefail

# Block new usage of legacy repository/path naming in tracked files.
# Historical archives under docs_tmp are intentionally excluded.

pattern="QueryLakeBackend"

# Allowed references:
# - retirement runbook documents the legacy alias explicitly.
# - repo migration doc captures historical rename details.
# - this guard script contains the pattern literal by design.
allowlist=(
  "docs/unification/symlink_retirement_runbook.md"
  "docs/unification/repo_migration.md"
  "scripts/ci_guard_legacy_querylakebackend_refs.sh"
)

pathspec=( "." ":(exclude)docs_tmp/**" )
for allowed in "${allowlist[@]}"; do
  pathspec+=( ":(exclude)${allowed}" )
done

set +e
matches="$(git grep -n "${pattern}" -- "${pathspec[@]}")"
status=$?
set -e

if [[ ${status} -eq 0 ]]; then
  echo "Legacy naming guard failed: found '${pattern}' references outside allowlist."
  echo
  echo "${matches}"
  exit 1
fi

if [[ ${status} -eq 1 ]]; then
  echo "Legacy naming guard passed: no disallowed '${pattern}' references found."
  exit 0
fi

echo "Legacy naming guard error: git grep exited with status ${status}."
exit ${status}
