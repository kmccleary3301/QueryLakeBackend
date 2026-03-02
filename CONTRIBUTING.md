# Contributing to QueryLakeBackend

This repository supports two main development tracks:

- Backend/runtime (`querylake-backend`)
- Python SDK (`querylake-sdk`)

Use this guide to keep contributions reviewable and release-safe.

## Baseline setup

From repo root:

```bash
cp .env.example .env
make bootstrap
```

If your change touches backend data paths or runtime:

```bash
make up-db
make run-api-only
make health
```

## Required checks before opening a PR

Run the smallest gate set that matches your change:

- SDK-only changes:
  - `make sdk-precommit-run`
  - `make sdk-ci`
- Retrieval/runtime/checks changes:
  - `make ci-unification`
  - `make ci-retrieval-smoke`
- Docs-only changes:
  - `make ci-docs`
- General backend code changes:
  - `make test`

If your change touches multiple areas, run all relevant gates.

## Change scope and PR hygiene

- Keep PRs focused on one functional objective.
- Include docs updates in the same PR for behavior/setup changes.
- Do not commit machine-local files or secrets (`.env`, local logs, private keys).
- If you alter CI scripts/workflows, run corresponding local commands before push.

## Retrieval and RAG-specific expectations

For changes that affect retrieval behavior, include:

- What changed (lane weights, parser behavior, fusion logic, constraints, etc.)
- What was measured (recall/MRR, overlap, latency, gate outputs)
- Which harness command(s) were run
- Any threshold updates and rationale

Use artifacts under `docs_tmp/RAG/` when appropriate for reproducibility.

## SDK expectations

For SDK changes:

- Preserve `QueryLakeClient.api()` compatibility.
- Prefer additive API changes over breaking changes.
- Keep CLI/profile behavior deterministic and documented.
- Ensure type/lint/tests pass via `make sdk-ci`.

## Commit message convention

Use concise scope-based prefixes when possible:

- `feat(sdk): ...`
- `fix(retrieval): ...`
- `chore(ci): ...`
- `docs(sdk): ...`

## Release notes

If your change affects SDK packaging or release flow, update:

- `docs/sdk/PYPI_RELEASE.md`
- `docs/sdk/API_REFERENCE.md` (if API/CLI changed)
