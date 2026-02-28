# PyPI Release Runbook (querylake-sdk)

This runbook publishes the standalone Python SDK located at `sdk/python`.

## Package scope

- Distribution: `querylake-sdk`
- Source root: `sdk/python/src/querylake_sdk`
- CLI entrypoint: `querylake`

## Pre-release checks

From repo root:

```bash
make sdk-test
make sdk-build
make sdk-release-check
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

```bash
cd sdk/python
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

For TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Or use the helper script from repo root:

```bash
# Check-only
./scripts/dev/release_sdk.sh check

# Upload to TestPyPI
./scripts/dev/release_sdk.sh testpypi

# Upload to PyPI
./scripts/dev/release_sdk.sh pypi
```

Or use GitHub Actions:

- Workflow: `.github/workflows/sdk_publish.yml`
- Trigger: manual (`workflow_dispatch`)
- Input `target`: `testpypi` or `pypi`

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
