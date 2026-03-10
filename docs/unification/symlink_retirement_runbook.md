# Legacy Path Symlink Retirement Runbook

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)
[![Legacy Path Guard](https://github.com/kmccleary3301/QueryLake/actions/workflows/legacy_path_guard.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/legacy_path_guard.yml)

Safe retirement process for the deprecated local `QueryLakeBackend` compatibility alias while preserving rollback clarity.

| Field | Value |
|---|---|
| Audience | Maintainers, CI owners, local environment operators |
| Use this when | Use this when checking retirement criteria, dated checkpoints, cutover steps, or rollback commands for the legacy local symlink. |
| Prerequisites | Awareness of the repo/path migration and the current shared-host setup. |
| Related docs | [`repo_migration.md`](repo_migration.md), [`repo_pinning_playbook.md`](repo_pinning_playbook.md), [`../setup/DEVELOPER_SETUP.md`](../setup/DEVELOPER_SETUP.md) |
| Status | 🟢 active runbook |

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

## T0 Checkpoint Record

- T0 date: `2026-03-05`
- Owner: `QueryLake maintainer (repo unification lead)`
- Status: `active`

### Evidence (merged changes)

- QueryLake legacy-name CI guard + runbook linkage:
  - PR: `https://github.com/kmccleary3301/QueryLake/pull/8`
  - Merge commit: `4eb59ecf560fa2ae0dc47398e7ed1f22290f17e7`
- QueryLake README topology restoration (post-merge correction):
  - PR: `https://github.com/kmccleary3301/QueryLake/pull/9`
  - Merge commit: `c742880e62581b0bd2731cfe7236517aa8e5c604`
- Breadboard downstream legacy-name guard:
  - PR: `https://github.com/kmccleary3301/breadboard/pull/20`
  - Merge commit: `0aff24f9f41c6fd532503f24b78f7b09fbdfb60f`
- Hermes downstream legacy-name guard (active `first_commit` line):
  - PR: `https://github.com/kmccleary3301/Hermes/pull/1`
  - Merge commit: `f3e963d6835a9aa9c875168b7b3e3dcf160aca41`

### Next dated checkpoints

- `2026-03-19` (`T0 + 14d`): strict enforcement review + violation sweep
- `2026-04-04` (`T0 + 30d`): remove alias from setup/bootstrap instructions
- `2026-04-19` (`T0 + 45d`): host-level symlink retirement cutover

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
