# PyPI Release Runbook (querylake-sdk)

This runbook publishes the standalone Python SDK located at `sdk/python`.

For CI profile definitions (light matrix, release guard, local parity), see:
`docs/sdk/CI_PROFILES.md`.

For guarded TestPyPI dry-run publishing, see:
`docs/sdk/TESTPYPI_DRYRUN.md`.

## Package scope

- Distribution: `querylake-sdk`
- Source root: `sdk/python/src/querylake_sdk`
- CLI entrypoint: `querylake`

## Pre-release checks

From repo root:

```bash
make sdk-precommit-run
make sdk-ci
make sdk-release-check
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main SKIP_REMOTE_CHECK=1
```

`make sdk-ci` runs the same quality gate used in CI (`scripts/ci_sdk_checks.sh`):
- lint (`ruff`)
- typing (`mypy`)
- SDK tests (with `dev` extra)
- wheel/sdist build
- `twine check`
- wheel content validation (`py.typed`, metadata tokens)

The lint/type/test portion is centralized in:

```bash
bash scripts/dev/sdk_quality_gate.sh all
```

Expected artifacts:

- `sdk/python/dist/querylake_sdk-<version>-py3-none-any.whl`
- `sdk/python/dist/querylake_sdk-<version>.tar.gz`

## Version bump

Edit:

- `sdk/python/pyproject.toml` -> `[project].version`

Use semver:

- Patch: docs/fixes only
- Minor: backward-compatible feature additions
- Major: breaking API changes

## Publish

Use the helper script from repo root:

```bash
# Check-only
./scripts/dev/release_sdk.sh check

# Validate publish guard directly (optional, explicit)
make sdk-publish-guard TARGET=testpypi GITHUB_REF=refs/heads/main

# Upload to TestPyPI
./scripts/dev/release_sdk.sh testpypi

# Upload to PyPI
./scripts/dev/release_sdk.sh pypi
```

Or use GitHub Actions:

- Workflow: `.github/workflows/sdk_publish.yml`
- Trigger: manual (`workflow_dispatch`)
- Input `target`: `testpypi` or `pypi`
- Guard policy:
  - `target=pypi` must run from `refs/heads/main` or tag `refs/tags/v<version>`
  - `target=pypi` requires stable semver version format `X.Y.Z`
  - target/version uniqueness is checked against the selected index

Dry-run TestPyPI workflow:

- Workflow: `.github/workflows/sdk_publish_dryrun.yml`
- Trigger: manual + nightly schedule
- Behavior: generates unique `dev` version, publishes to TestPyPI, verifies clean install

## Post-release verification

```bash
python -m venv /tmp/ql-sdk-verify
source /tmp/ql-sdk-verify/bin/activate
pip install querylake-sdk
querylake --help
```

## Backward compatibility policy

- Keep `QueryLakeClient.api()` stable so researchers can call new backend functions without SDK blocking.
- New helper methods should be additive.
- Deprecations must be documented in `README.md` + release notes before removal.
