# SDK CI Profiles and Publish Policy

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![SDK Publish Dry-Run](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_publish_dryrun.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_publish_dryrun.yml)

Defines the standard CI profiles, guardrails, and release policy for `querylake-sdk`.

| Field | Value |
|---|---|
| Audience | Maintainers, release managers, and engineers modifying CI or publish policy |
| Use this when | Use this when changing SDK CI, release requirements, branch policy, or runtime/cost governance. |
| Prerequisites | Basic familiarity with the SDK package layout and GitHub Actions workflows. |
| Related docs | [`PYPI_RELEASE.md`](PYPI_RELEASE.md), [`TESTPYPI_DRYRUN.md`](TESTPYPI_DRYRUN.md), [`CI_PERFORMANCE_POLICY.md`](CI_PERFORMANCE_POLICY.md) |
| Status | 🟢 maintained policy document |

This document defines the standard CI profiles and release guardrails for `querylake-sdk`.

## CI profiles

### 1) Local fast profile (developer loop)

Use when iterating quickly on SDK code:

```bash
make sdk-lint
make sdk-type
make sdk-test
```

### 2) Local release profile (pre-publish parity)

Use before merging release-related changes or publishing:

```bash
make sdk-ci
make sdk-release-check
```

This executes the same checks used in CI release guard:

- lint + type + tests
- wheel/sdist build
- metadata verification (`twine check`)
- wheel contract checks (`scripts/dev/verify_sdk_wheel.py`)

### 3) GitHub light matrix profile (`sdk_checks.yml`)

Workflow: `.github/workflows/sdk_checks.yml`

- `sdk-lint-type`: `make sdk-lint` + `make sdk-type` on Python `3.12`
- `sdk-light-matrix`: `make sdk-test` on Python `3.10`, `3.11`, `3.12` (tests-only matrix)
- `sdk-release-guard`: guard script contract + single package build + `twine check` + wheel verification

This layout removes duplicated full-quality runs while preserving all gates:
- lint/type runs once
- tests run per-Python
- build/metadata/wheel checks run once

### 4) TestPyPI dry-run publish profile (`sdk_publish_dryrun.yml`)

Workflow: `.github/workflows/sdk_publish_dryrun.yml`

- runs manual + nightly
- enforces main-branch default policy
- generates unique `dev` version per run
- validates publish guard + full SDK CI checks
- uploads to TestPyPI
- verifies clean install/import/CLI/offline demo

### 5) Runtime profiling/governance profile (`ci_runtime_profiler.yml`)

Workflow: `.github/workflows/ci_runtime_profiler.yml`

- runs daily + manual
- builds 7-day runtime/cost profile from Actions run metadata
- writes JSON + markdown artifacts
- optionally computes a delta against committed baseline

### 6) Live staging integration profile (`sdk_live_integration.yml`)

Workflow: `.github/workflows/sdk_live_integration.yml`

- runs manual only
- enforces preflight environment contract before execution
- defaults to read-only checks
- optional manual write-path smoke with explicit enable switch
- explicitly non-blocking for merge CI (release-confidence lane only)

## Trigger and cost governance

### Trigger policy

- `sdk_checks.yml`:
  - runs on PR/push for SDK-relevant paths only
  - provides default quality gate signal
  - applies uv cache keyed by SDK dependency file
- `sdk_publish_dryrun.yml`:
  - nightly + manual dispatch
  - restricted to `main` by default (explicit non-main override available for trials)
  - applies uv cache keyed by SDK dependency file
- `sdk_live_integration.yml`:
  - manual read-only by default
  - manual write-path only when `allow_write=true`
  - applies uv cache keyed by SDK dependency file
- `retrieval_eval.yml`:
  - PR/push triggers are path-filtered to retrieval/search/ingestion surfaces
  - heavy profile remains dispatch-only
- `ci_runtime_profiler.yml`:
  - nightly + manual
  - used for governance trend and regression detection
  - supports configurable p95/compute regression thresholds and optional fail-on-regression

### Weekly runtime/cost budget targets

- `sdk_checks.yml`: target median <= 8 min, p95 <= 12 min
- `sdk_publish_dryrun.yml`: target median <= 15 min, p95 <= 20 min
- `sdk_live_integration.yml`: target median <= 12 min, p95 <= 18 min
- aggregate SDK CI compute budget: <= 550 runner-minutes/week

### Escalation policy

Escalate to maintainer review when any of the following occurs for two consecutive days:

1. p95 runtime regression > 15% versus baseline.
2. workflow failure-rate > 10% for non-code reasons (infra/index/auth).
3. weekly compute budget exceeded by > 20%.

Required remediation output:

- root-cause summary
- reverted/adjusted workflow plan
- expected rollback and verification window

### Required-check stance

- Merge-required quality gates: `sdk_checks.yml` (lint/type/test/release-guard) and policy checks tied to code quality.
- Optional confidence lanes: `sdk_live_integration.yml` (manual live staging), retrieval heavy/nightly suites.
- Rationale: avoid coupling PR throughput to external staging/network/secret availability while preserving a strong pre-release validation path.

## Publish policy

Workflow: `.github/workflows/sdk_publish.yml`

Before any upload, `scripts/dev/verify_sdk_publish_guard.py` enforces:

- `target=pypi` requires:
  - stable semver `X.Y.Z` in `sdk/python/pyproject.toml`
  - ref is `refs/heads/main` or exact tag `refs/tags/v<version>`
- target/version uniqueness against selected index (PyPI or TestPyPI)

Then it runs full `scripts/ci_sdk_checks.sh` before upload.

## Local guard usage

Explicit guard check:

```bash
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main
make sdk-publish-guard TARGET=pypi GITHUB_REF=refs/tags/v0.1.0
```

Optional skip for remote index lookup (useful for isolated/offline checks):

```bash
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main SKIP_REMOTE_CHECK=1
```

## Migration notes

### Old flow

Ad-hoc publish with implicit assumptions:

```bash
bash scripts/ci_sdk_checks.sh
./scripts/dev/release_sdk.sh testpypi
```

### New flow

Guarded publish with explicit constraints:

```bash
make sdk-ci
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main
./scripts/dev/release_sdk.sh testpypi
```

This prevents accidental PyPI publishes from feature branches and blocks duplicate versions.