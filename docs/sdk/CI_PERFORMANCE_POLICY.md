# CI Performance Policy (SDK + Core Workflows)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![CI Runtime Profiler](https://github.com/kmccleary3301/QueryLake/actions/workflows/ci_runtime_profiler.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/ci_runtime_profiler.yml)

Governance policy for CI runtime, cost, and regression thresholds across SDK-adjacent workflows.

| Field | Value |
|---|---|
| Audience | Maintainers and CI/operators responsible for keeping validation useful without wasting runner time |
| Use this when | Use this when adding CI steps, evaluating runtime regressions, or reviewing workflow cost/performance drift. |
| Prerequisites | Familiarity with the SDK CI profiles and the GitHub Actions workflow set. |
| Related docs | [`CI_PROFILES.md`](CI_PROFILES.md) |
| Status | 🟢 maintained policy document |

This document defines how QueryLake measures and governs CI runtime/cost for SDK and related workflows.

## Scope

- SDK checks and release workflows
- docs/unification checks
- retrieval smoke/eval workflows

## Baseline cadence

- Generate a 7-day profile daily via:
  - `.github/workflows/ci_runtime_profiler.yml`
- Keep artifacts for trend review and regression triage.

## Profile metrics

Collected per workflow and overall:

- run count
- success rate
- error rate
- rerun rate (`run_attempt > 1`)
- duration (median / p95 / mean)
- queue time (median / p95 / mean)
- total compute minutes (duration-sum / 60)

## Regression gates (initial policy)

When comparing `before -> after`:

- overall duration p95 must not regress by more than `15%`
- overall compute minutes must not regress by more than `20%`

Tools:

- `scripts/dev/ci_runtime_profile.py`
- `scripts/dev/ci_runtime_delta.py`

## Local usage

```bash
# Generate profile from local JSON payload (offline test mode)
python scripts/dev/ci_runtime_profile.py \
  --repo owner/repo \
  --input-runs-json /tmp/workflow_runs.json \
  --out-json /tmp/ci_profile.json \
  --out-md /tmp/ci_profile.md

# Compare two profiles
python scripts/dev/ci_runtime_delta.py \
  --before-json /tmp/ci_profile_before.json \
  --after-json /tmp/ci_profile_after.json \
  --out-md /tmp/ci_profile_delta.md \
  --fail-on-regression
```

Make wrappers:

```bash
make ci-runtime-profile REPO=owner/repo DAYS=7
make ci-runtime-delta BEFORE=/tmp/ci_before.json AFTER=/tmp/ci_after.json
```

## Workflow policy

1. Keep light PR profile fast:
- matrix tests across supported Python versions
- single-pass lint/type

2. Keep heavy checks explicit:
- release/publish guard paths
- retrieval heavy tracks outside default PR path

3. Any new CI step must include:
- purpose
- expected runtime impact
- why it cannot be merged into an existing step

## Review checklist for CI changes

- Does this run on every PR push, or only on relevant path filters?
- Can this step reuse existing artifacts/caches?
- Is this duplicated in another job?
- Is failure signal unique and actionable?
- What is expected p95 runtime impact?