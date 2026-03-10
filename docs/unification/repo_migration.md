# Repository Migration: Frontend Rename + Monorepo Import

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)
[![Legacy Path Guard](https://github.com/kmccleary3301/QueryLake/actions/workflows/legacy_path_guard.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/legacy_path_guard.yml)

Canonical record of the frontend rename, monorepo import, and local path naming transition to the `QueryLake` repo identity.

| Field | Value |
|---|---|
| Audience | Contributors, release maintainers, downstream repo maintainers |
| Use this when | Use this when you need the definitive repo-name/path history or need to explain why `QueryLake` is the canonical monorepo. |
| Prerequisites | Basic familiarity with the old `QueryLakeBackend` / `QueryLakeStudio` split. |
| Related docs | [`symlink_retirement_runbook.md`](symlink_retirement_runbook.md), [`repo_pinning_playbook.md`](repo_pinning_playbook.md), [`../setup/DEVELOPER_SETUP.md`](../setup/DEVELOPER_SETUP.md) |
| Status | 🟢 active migration record |

## Summary

This migration consolidates QueryLake development around the backend repository while preserving frontend history.

- Frontend repo renamed:
  - from `kmccleary3301/QueryLake`
  - to `kmccleary3301/QueryLakeStudio`
- Frontend repo status:
  - marked **deprecated** via README update
  - points contributors to the backend repository
- Frontend code imported into backend repository:
  - path: `apps/studio/`
  - method: `git subtree` (history-preserving)
- Backend repository renamed:
  - from `kmccleary3301/QueryLakeBackend`
  - to `kmccleary3301/QueryLake`

## Why this migration

- Backend is the platform core (runtime, APIs, ingestion, retrieval, SDK).
- Frontend and backend versioning are easier to coordinate in one repository.
- Preserving frontend history keeps blame/audit trails intact.

## Implementation details

### GitHub rename

Frontend repository was renamed in place on GitHub:

```bash
gh api -X PATCH repos/kmccleary3301/QueryLake -f name='QueryLakeStudio'
```

### Deprecation commit on frontend repo

A README-only commit was added to the renamed frontend repo:

- Commit: `97fa7a8dd5c71096b455d62154150366315ff9f3`
- Message: `docs: mark QueryLakeStudio as deprecated and point to main repo`

### Monorepo import command

Frontend source was imported into backend with full commit history:

```bash
git remote add studio https://github.com/kmccleary3301/QueryLakeStudio.git
git fetch studio master
git subtree add --prefix=apps/studio studio/master -m "monorepo: import QueryLakeStudio into apps/studio"
```

## Post-migration operating model

- New frontend changes should land in `apps/studio/` in backend repo.
- `QueryLakeStudio` remains available as an archive/deprecation pointer.
- CI, release notes, and docs should treat backend repo as the canonical development surface.

## Current naming state

- `QueryLake`: canonical monorepo (backend + SDK + studio at `apps/studio`)
- `QueryLakeStudio`: deprecated archive/deprecation pointer

## Local path transition status

- Canonical local path:
  - `/shared_folders/querylake_server/QueryLake`
- Deprecated compatibility alias:
  - `/shared_folders/querylake_server/QueryLakeBackend`
- Retirement policy:
  - `docs/unification/symlink_retirement_runbook.md`
