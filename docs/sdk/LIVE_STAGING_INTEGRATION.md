# Live Staging Integration Plan and Contract

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![SDK Live Integration](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_live_integration.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_live_integration.yml)

Contract and safety policy for running SDK integration tests against a live QueryLake deployment.

| Field | Value |
|---|---|
| Audience | Maintainers, release operators, and engineers validating staging behavior |
| Use this when | Use this when preparing or running manual live integration checks against a staging environment. |
| Prerequisites | A staging deployment, GitHub environment variables/secrets, and understanding of read-only vs write-path safety controls. |
| Related docs | [`CI_PROFILES.md`](CI_PROFILES.md), [`PYPI_RELEASE.md`](PYPI_RELEASE.md) |
| Status | 🟢 maintained integration contract |

This document defines prerequisites, safety controls, and execution policy for live SDK integration tests.

## Purpose

Validate SDK + CLI behavior against a real QueryLake deployment while preventing accidental production-side effects.

## Workflow

- File: `.github/workflows/sdk_live_integration.yml`
- Trigger:
  - manual (`workflow_dispatch`) only
- Test marker: `integration_live`
- Test path: `sdk/python/tests/integration/`

## Required environment contract

Set these in GitHub repository vars/secrets:

- `vars.QUERYLAKE_LIVE_BASE_URL` (required)
- `secrets.QUERYLAKE_LIVE_OAUTH2` or `secrets.QUERYLAKE_LIVE_API_KEY` (one required)
- `vars.QUERYLAKE_LIVE_TEST_COLLECTION_ID` (required for search and optional write smoke)
- `vars.QUERYLAKE_LIVE_TEST_QUERY` (optional; defaults in test)
- `vars.QUERYLAKE_LIVE_ALLOW_NON_STAGING` (optional; `1` to bypass host safety gate)
- `vars.QUERYLAKE_LIVE_TIMEOUT_SECONDS` (optional; defaults to `20`)
- `vars.QUERYLAKE_LIVE_RETRY_ATTEMPTS` (optional; defaults to `3`)
- `vars.QUERYLAKE_LIVE_RETRY_DELAY_SECONDS` (optional; defaults to `1`)
- `vars.QUERYLAKE_LIVE_STRICT_EXPECTATIONS` (optional; `1` enables deterministic term assertions)
- `vars.QUERYLAKE_LIVE_QUERY_CASES_PATH` (optional; defaults to `sdk/python/tests/integration/live_query_cases.json`)

Manual input:

- `allow_write`:
  - `false` (default): read-only health/search checks
  - `true`: includes upload/delete smoke against test collection

## Safety controls

Preflight script: `scripts/dev/live_integration_preflight.py`

Enforces:

- base URL must exist
- auth token/key must exist
- placeholder values (e.g. `changeme`, `example`) are rejected
- non-staging hosts blocked by default
- write mode requires explicit test collection id

## Test coverage (current scaffold)

1. Health/readiness/models smoke
2. Hybrid search smoke on test collection
3. Optional write-path smoke:
   - upload one document
   - delete the uploaded document in teardown
4. Deterministic hybrid assertions (strict mode):
   - run canonical query cases from fixture JSON
   - enforce min row counts per case
   - optionally enforce expected term hits in returned rows

## Fail-closed behavior

- Missing required env/auth -> workflow fails before tests run.
- Suspicious host (non-staging) -> workflow fails unless explicit override.
- Write path disabled by default.

## Operational policy

- This workflow is intentionally **non-blocking** and **manual-only** for CI.
- Use it as a pre-release confidence check, not a merge gate.
- Write-path runs only on manual dispatch with explicit `allow_write=true`.
- Any failure should include junit artifact and step summary for triage.
- Integration tests emit `live_metrics.jsonl` with per-operation latency and retry outcomes.
- Integration tests include best-effort request-id capture in `live_metrics.jsonl`.

## Deterministic corpus and query fixture

Fixture file:

- `sdk/python/tests/integration/live_query_cases.json`

Bootstrap an isolated integration collection (recommended once per environment):

```bash
uv run --project sdk/python \
  python scripts/dev/live_integration_bootstrap.py \
  --collection-title "SDK Live Integration Fixture"
```

Bootstrap output contract:

- `docs_tmp/RAG/ci/live_integration/bootstrap_contract.json`
- includes recommended values for:
  - `QUERYLAKE_LIVE_TEST_COLLECTION_ID`
  - `QUERYLAKE_LIVE_QUERY_CASES_PATH`
  - `QUERYLAKE_LIVE_STRICT_EXPECTATIONS=1`

It defines:

- `documents`: canonical documents for deterministic live seeding
- `cases`: canonical retrieval queries and minimum expected rows

Seed the fixture corpus (manual, authenticated):

```bash
uv run --project sdk/python \
  python scripts/dev/live_integration_seed_fixture.py \
  --collection-id "$QUERYLAKE_LIVE_TEST_COLLECTION_ID" \
  --create-embeddings \
  --create-sparse-embeddings \
  --await-embedding
```

Seed output manifest:

- `docs_tmp/RAG/ci/live_integration/seed_fixture_results.json`

## Next extension points

- Add collection lifecycle checks (create/modify/list + cleanup) once dedicated test tenant policy is finalized.
- Add lane-specific retrieval parity assertions tied to known seeded corpus and capture lane-wise top-k drift trends.
- Add run-level latency SLO thresholds and flaky-test tracking.

## Observed failure modes and mitigations (2026-03-03)

1. Missing credentials or base URL in workflow environment.
   - symptom: preflight exits before tests
   - mitigation: required vars/secrets contract, fail-closed before test execution

2. Non-staging host accidentally targeted.
   - symptom: preflight rejects host unless override is explicitly enabled
   - mitigation: keep `QUERYLAKE_LIVE_ALLOW_NON_STAGING` unset by default

3. Write-path run requested without test collection id.
   - symptom: preflight hard-fails in write mode
   - mitigation: require `QUERYLAKE_LIVE_TEST_COLLECTION_ID` whenever `allow_write=true`

4. Transient staging/API instability during nightly/manual runs.
   - symptom: intermittent integration failure
   - mitigation: bounded retry/backoff in integration tests + retain `junit.xml` and `live_metrics.jsonl` artifacts for triage

5. Repository-level live vars/secrets absent on branch validation runs.
   - symptom: preflight fails with `QUERYLAKE_LIVE_BASE_URL is required` before tests start
   - evidence: workflow run `22642371545` (`docs_tmp/RAG/ci/live_integration/run_22642371545_failed.log`)
   - mitigation: define the required vars/secrets contract before manual live checks