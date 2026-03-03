# Live Staging Integration Plan and Contract

This document defines prerequisites, safety controls, and execution policy for live SDK integration tests.

## Purpose

Validate SDK + CLI behavior against a real QueryLake deployment while preventing accidental production-side effects.

## Workflow

- File: `.github/workflows/sdk_live_integration.yml`
- Trigger:
  - manual (`workflow_dispatch`)
  - nightly schedule
- Test marker: `integration_live`
- Test path: `sdk/python/tests/integration/`

## Required environment contract

Set these in GitHub repository vars/secrets:

- `vars.QUERYLAKE_LIVE_BASE_URL` (required)
- `secrets.QUERYLAKE_LIVE_OAUTH2` or `secrets.QUERYLAKE_LIVE_API_KEY` (one required)
- `vars.QUERYLAKE_LIVE_TEST_COLLECTION_ID` (required for search and optional write smoke)
- `vars.QUERYLAKE_LIVE_TEST_QUERY` (optional; defaults in test)
- `vars.QUERYLAKE_LIVE_ALLOW_NON_STAGING` (optional; `1` to bypass host safety gate)

Manual input:

- `allow_write`:
  - `false` (default): read-only health/search checks
  - `true`: includes upload/delete smoke against test collection

## Safety controls

Preflight script: `scripts/dev/live_integration_preflight.py`

Enforces:

- base URL must exist
- auth token/key must exist
- non-staging hosts blocked by default
- write mode requires explicit test collection id

## Test coverage (current scaffold)

1. Health/readiness/models smoke
2. Hybrid search smoke on test collection
3. Optional write-path smoke:
   - upload one document
   - delete the uploaded document in teardown

## Fail-closed behavior

- Missing required env/auth -> workflow fails before tests run.
- Suspicious host (non-staging) -> workflow fails unless explicit override.
- Write path disabled by default.

## Operational policy

- Nightly run in read-only mode.
- Write-path runs only on manual dispatch with explicit `allow_write=true`.
- Any failure should include junit artifact and step summary for triage.

## Next extension points

- Add collection lifecycle checks (create/modify/list + cleanup) once dedicated test tenant policy is finalized.
- Add lane-specific retrieval parity assertions tied to known seeded corpus.
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
   - mitigation: retain `junit.xml` + artifact diagnostics and use explicit rerun triage policy
