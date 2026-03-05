# Legacy Path Symlink Retirement Runbook

## Purpose

Define a safe, explicit process to retire the legacy local path alias:

- legacy alias: `/shared_folders/querylake_server/QueryLakeBackend`
- canonical path: `/shared_folders/querylake_server/QueryLake`

The goal is to finish repository naming unification without breaking local automation, CI scripts, or downstream repos that still depend on the legacy alias.

## Current State

- The legacy path is currently a symlink:
  - `/shared_folders/querylake_server/QueryLakeBackend -> /shared_folders/querylake_server/QueryLake`
- Canonical Git remote and project identity are already `QueryLake`.
- Downstream CI guards were added in Breadboard and Hermes to block new legacy-name references in active branches.
- QueryLake CI now enforces a legacy-name guard via:
  - `scripts/ci_guard_legacy_querylakebackend_refs.sh`
  - `scripts/ci_unification_checks.sh`
  - `.github/workflows/unification_checks.yml`

## Retirement Criteria (all required)

1. **No active code references** to `QueryLakeBackend` in:
   - QueryLake tracked source/docs/scripts (excluding `docs_tmp/` historical artifacts).
   - Active branches in Breadboard and Hermes.
2. **CI green** on unification checks for at least 7 consecutive days after criteria (1) is true.
3. **Setup docs updated** so new users only see `QueryLake` canonical path examples.
4. **Fallback tested**: recreate symlink in one command and validate startup/tests.

## Timeline (relative to approval date `T0`)

1. **T0 (approval + merge of this runbook)**
   - Announce deprecation window in release notes / team channel.
   - Mark alias as deprecated in setup docs.
2. **T0 + 14 days**
   - Enable strict CI guard in QueryLake for new legacy references (scoped to tracked code/docs, excluding `docs_tmp/`).
   - Fix any remaining violations.
3. **T0 + 30 days**
   - Remove symlink from developer setup scripts and local bootstrap instructions.
   - Keep manual rollback command documented.
4. **T0 + 45 days (retirement cutover)**
   - Remove symlink from shared environment hosts.
   - Run smoke checks from canonical path only.

## Execution Checklist

1. Verify symlink status:
   - `python -c "import os; print(os.path.islink('/shared_folders/querylake_server/QueryLakeBackend'))"`
2. Scan for active legacy refs (exclude historical archives):
   - `rg -n "QueryLakeBackend" --glob '!docs_tmp/**'`
3. Validate CI:
   - `make ci-unification`
4. Remove symlink on cutover host:
   - `rm /shared_folders/querylake_server/QueryLakeBackend`
5. Validate canonical path flows:
   - repo open, startup command, smoke tests, SDK docs examples.

## Rollback (one-command restore)

If any workflow still requires the alias after cutover:

```bash
ln -s /shared_folders/querylake_server/QueryLake /shared_folders/querylake_server/QueryLakeBackend
```

Then re-run the failing workflow and file a follow-up issue to remove the stale dependency.

## Ownership

- Owner: QueryLake maintainer (repo unification lead)
- Review cadence: once per week until retirement cutover is complete
- Artifact of record: this document + `docs/unification/compat_matrix.md`
