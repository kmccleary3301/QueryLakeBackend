# SDK CI Profiles and Publish Policy

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

- `sdk-light-matrix`: `make sdk-test` on Python `3.10`, `3.11`, `3.12`
- `sdk-lint-type`: `make sdk-lint` + `make sdk-type` on Python `3.12`
- `sdk-release-guard`: runs `scripts/dev/release_sdk.sh check` after both jobs pass

The matrix validates interpreter compatibility while avoiding duplicated lint/type cost.

### 4) TestPyPI dry-run publish profile (`sdk_publish_dryrun.yml`)

Workflow: `.github/workflows/sdk_publish_dryrun.yml`

- runs manual + nightly
- enforces main-branch default policy
- generates unique `dev` version per run
- validates publish guard + full SDK CI checks
- uploads to TestPyPI
- verifies clean install/import/CLI/offline demo

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
