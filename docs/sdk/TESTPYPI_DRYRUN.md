# TestPyPI Dry-Run Publish Runbook

This runbook covers guarded dry-run publishing of `querylake-sdk` to TestPyPI.

## Purpose

Validate the full release path (guard checks, build, upload, install verification) without touching production PyPI.

## Workflow

- File: `.github/workflows/sdk_publish_dryrun.yml`
- Trigger:
  - manual (`workflow_dispatch`)
  - scheduled nightly
- Target: TestPyPI only

## Guardrails

The workflow enforces:

- Branch policy: `main` only by default (manual override available)
- Deterministic unique dry-run version generated from base semver:
  - `X.Y.Z.dev<run_id><run_attempt>`
- Publish guard checks before upload:
  - index target validation
  - version uniqueness check against TestPyPI

## End-to-end flow

1. Compute/write dry-run version into `sdk/python/pyproject.toml` (CI workspace only).
2. Run guard checks via `scripts/dev/verify_sdk_publish_guard.py`.
3. Run full SDK CI checks (`scripts/ci_sdk_checks.sh`).
4. Upload to TestPyPI.
5. Create clean venv and install exact published version from TestPyPI.
6. Verify:
   - `import querylake_sdk`
   - `querylake --help`
   - offline SDK example (`examples/sdk/rag_bulk_ingest_and_search.py --offline-demo`)

## Artifacts and reporting

The workflow uploads:

- built wheel/sdist
- dry-run version metadata
- CLI help output
- offline demo outputs
- structured JSON summary

It also emits a GitHub step summary with published version and package URL.

## Failure and recovery guide

### Case 1: Branch policy failure

Symptom:
- workflow fails on branch policy step

Action:
- run from `main`, or explicitly set `allow_non_main=true` for manual trial runs.

### Case 2: Duplicate version on TestPyPI

Symptom:
- publish guard fails uniqueness check

Action:
- rerun workflow (new run id/attempt gives a fresh `dev` suffix), or dispatch a new run.

### Case 3: Publish succeeds but install verification fails

Symptom:
- venv install/import/CLI check fails

Action:
- inspect uploaded artifacts (`summary.json`, CLI output, dist files).
- confirm dependency resolution and package metadata.
- rerun after fixing package/build issues.

### Case 4: TestPyPI transient outage or timeout

Symptom:
- upload or install step fails with network/index errors

Action:
- rerun workflow.
- if repeated, temporarily disable scheduled runs and open infra issue.

## Operator recovery playbook

### Branch-policy reject

```bash
gh workflow run sdk_publish_dryrun.yml --ref main -f allow_non_main=false
```

### Intentional non-main validation

```bash
gh workflow run sdk_publish_dryrun.yml --ref <branch> -f allow_non_main=true
```

### Guard-only validation (local)

```bash
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main
```

### Trusted publisher failure remediation checklist

1. Confirm GitHub environment is exactly `testpypi`.
2. Confirm TestPyPI trusted publisher matches:
   - repository: `kmccleary3301/QueryLake`
   - workflow file: `.github/workflows/sdk_publish_dryrun.yml`
   - branch/ref policy expected by TestPyPI
   - environment claim: `testpypi`
3. Re-run dry-run workflow and confirm publish stage progresses.

### Fast evidence capture for a run

```bash
RUN_ID=<actions_run_id>
gh run view "$RUN_ID" --json databaseId,name,event,headBranch,status,conclusion,createdAt,updatedAt,url
gh run view "$RUN_ID" --log > docs_tmp/RAG/ci/sdk_publish_dryrun/run_${RUN_ID}.log
```

## Promotion handoff to production PyPI

Dry-run completion is required but not sufficient for production release. Before `sdk_publish.yml` to `pypi`:

1. Dry-run evidence bundle present (run URL + log + artifact summary).
2. Trusted publisher wiring verified on both TestPyPI and PyPI environments.
3. Release version is stable semver (`X.Y.Z`) and not pre-release.
4. `main` or matching `refs/tags/vX.Y.Z` ref policy is satisfied.
5. Final `scripts/ci_sdk_checks.sh` result is green on release commit.
## Local helper commands

```bash
# Generate a deterministic dry-run version (no file mutation)
make sdk-dryrun-version TOKEN=20260303180000

# Write dry-run version into pyproject.toml
make sdk-dryrun-version TOKEN=20260303180000 WRITE=1

# Validate publish target constraints locally
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main
```
