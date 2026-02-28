# QueryLake Developer Experience Plan (SDK + Setup + Docs)

This plan tracks the end-to-end work required to make QueryLake easier for developers and researchers, especially for RAG workflows.

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

## Outcomes

- QueryLake now has a dedicated lightweight SDK package with a practical CLI path.
- Backend setup is now scriptable and reproducible with explicit make targets.
- Docs are structured for both first-time developers and advanced RAG researchers.
