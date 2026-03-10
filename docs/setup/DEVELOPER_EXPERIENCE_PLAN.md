# QueryLake Developer Experience Plan (SDK + Setup + Docs)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)

Structured plan for making QueryLake easier to install, use, package, and extend for developers and researchers.

| Field | Value |
|---|---|
| Audience | Maintainers, SDK owners, developer-experience contributors |
| Use this when | Use this when you need the scope, phase breakdown, and remaining closeout work for the current DX overhaul. |
| Prerequisites | General familiarity with QueryLake setup pain points, SDK work, and docs/release goals. |
| Related docs | [`DEVELOPER_SETUP.md`](DEVELOPER_SETUP.md), [`../sdk/SDK_QUICKSTART.md`](../sdk/SDK_QUICKSTART.md), [`../README.md`](../README.md) |
| Status | 🟡 active plan with minor closeout items |

## Goals

1. Standardized SDK surface for backend functions and OpenAI-compatible routes.
2. Minimal-friction local setup for backend contributors.
3. Clear, polished docs that support both research iteration and production onboarding.
4. PyPI-ready packaging for a standalone SDK distribution.

## Execution phases

### Phase 1: SDK standardization

- [x] Define stable Python client interfaces (sync + async).
- [x] Implement robust error model (transport, HTTP status, API contract failures).
- [x] Add RAG helpers for collection/document/search workflows.
- [x] Preserve escape hatch (`client.api`) for full backend function reach.

### Phase 2: CLI and auth bootstrap

- [x] Add `querylake` CLI for profile-backed auth and health checks.
- [x] Add profile persistence under `~/.querylake/sdk_profiles.json`.
- [x] Add RAG quick commands: create-collection, upload, search.

### Phase 3: Packaging and release readiness

- [x] Create standalone SDK package at `sdk/python`.
- [x] Add SDK `pyproject.toml` with metadata and entrypoint.
- [x] Validate wheel/sdist build output.
- [x] Document release process for PyPI/TestPyPI.

### Phase 4: Setup streamlining

- [x] Add bootstrap script (`scripts/dev/bootstrap.sh`).
- [x] Add `.env.example` baseline.
- [x] Add task-centric `Makefile` targets for local setup/run/test/build.

### Phase 5: Documentation overhaul

- [x] Rewrite root `README.md` for developer-first onboarding.
- [x] Add focused setup guide and SDK quickstart.
- [x] Add SDK API reference and docs index.
- [x] Link package/release workflows clearly.

### Phase 6: Validation

- [x] Add SDK unit tests (client behavior + CLI behavior).
- [x] Run SDK tests.
- [x] Build SDK distribution artifacts.

### Phase 7: Bulk ingest durability

- [x] Add resumable checkpointed directory ingest for large corpus uploads.
- [x] Add deterministic upload selection hashing and checkpoint snapshots.
- [x] Add dry-run preflight outputs for ingestion planning.

### Phase 8: Ingest reproducibility controls

- [x] Add content-hash dedupe controls (none/per-run/collection/global).
- [x] Add idempotency strategies and prefix controls.
- [x] Add strict ingest profiles and profile-file overrides.

### Phase 9: CI profile hardening

- [x] Split SDK checks into light Python-version matrix + single-pass lint/type.
- [x] Gate release checks on matrix + lint/type pass.
- [x] Keep wheel build/metadata guard in release-check path.

### Phase 10: Publish safety rails

- [x] Add explicit publish guard script for target/version/ref constraints.
- [x] Enforce guard in `sdk_publish` workflow.
- [x] Integrate guard into local `release_sdk.sh` target flows.
- [x] Add automated tests for guard behavior.

### Phase 11: Docs IA and migration path

- [x] Add dedicated CI profiles and publish policy documentation.
- [x] Update release runbook to include guard workflow and branch/tag policy.
- [x] Add make target for explicit publish guard local validation.

### Phase 12: Consolidation

- [ ] Run full local gate suite (`sdk`, docs, unification, retrieval smoke).
- [ ] Publish final completion summary with current state and next headroom items.

## Outcomes

- QueryLake now has a dedicated lightweight SDK package with a practical CLI path.
- Backend setup is now scriptable and reproducible with explicit make targets.
- Docs are structured for both first-time developers and advanced RAG researchers.
- CI and release workflows now enforce explicit publish safety rules for PyPI/TestPyPI.
