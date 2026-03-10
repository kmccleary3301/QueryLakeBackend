# QueryLake Python SDK

[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![SDK Live Integration](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_live_integration.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_live_integration.yml)
[![SDK Publish Dry-Run](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_publish_dryrun.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_publish_dryrun.yml)
[![PyPI](https://img.shields.io/pypi/v/querylake-sdk?logo=pypi&color=F9A03C)](https://pypi.org/project/querylake-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/querylake-sdk?logo=python&color=3776AB)](https://pypi.org/project/querylake-sdk/)
[![License](https://img.shields.io/github/license/kmccleary3301/QueryLake?color=2ea44f)](../../LICENSE)

Official lightweight Python SDK for QueryLake.

`querylake-sdk` is the intended integration surface for most application developers, researchers, and automation users. It wraps the common auth, collection, ingestion, retrieval, and health-check flows so client code does not have to manually reason about QueryLake’s route topology or session mechanics.

## Table of contents

- [What the SDK covers](#what-the-sdk-covers)
- [Install](#install)
- [Quickstart](#quickstart)
- [CLI usage](#cli-usage)
- [Python usage](#python-usage)
- [Offline examples](#offline-examples)
- [Profile management](#profile-management)
- [Async client](#async-client)
- [Compatibility and scope](#compatibility-and-scope)
- [Documentation map](#documentation-map)
- [Development and release](#development-and-release)

## What the SDK covers

| Surface | Scope | Status |
|---|---|---|
| Sync client | collections, documents, upload, hybrid search, health | 🟢 |
| Async client | async auth and health flows, low-friction async integration | 🟢 |
| CLI | login profiles, health checks, collection and RAG workflows | 🟢 |
| Typed helpers | typed result rows and option objects for common workflows | 🟢 |
| Raw escape hatch | `client.api("function_name", payload)` | 🟢 |
| Live staging integration contract | GitHub Actions workflow + integration tests | 🟢 |

### Why use the SDK instead of calling the backend directly?

- it standardizes auth/profile handling,
- it keeps low-level route details out of app code,
- it provides a stable developer-facing surface while backend internals evolve,
- and it gives you a CLI that matches the Python client’s core workflows.

> If you are building applications against QueryLake, start here first. Only drop down to backend internals if you are actively changing the runtime itself.

## Install

### From PyPI

```bash
pip install querylake-sdk
```

### For local development from this repository

```bash
cd sdk/python
pip install -e ".[dev]"
```

### Quick environment check

```bash
querylake --help
querylake --url http://127.0.0.1:8000 doctor
```

## Quickstart

### Five-minute path

```bash
pip install querylake-sdk
querylake --url http://127.0.0.1:8000 doctor
querylake setup --url http://127.0.0.1:8000 --profile local --username demo --password demo-pass --non-interactive
querylake login --url http://127.0.0.1:8000 --profile local --username demo --password demo-pass
querylake --profile local rag create-collection --name "papers"
querylake --profile local rag upload --collection-id <id> --file ./paper.pdf --await-embedding
querylake --profile local rag search --collection-id <id> --query "hybrid retrieval design" --preset tri-lane --with-metrics
```

### Quickstart in Python

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

collection = client.create_collection(name="sdk-demo")
collection_id = collection["hash_id"]

client.upload_document(
    file_path="paper.pdf",
    collection_hash_id=collection_id,
    await_embedding=True,
    create_sparse_embeddings=True,
)

rows = client.search_hybrid_chunks(
    query="What is the main contribution?",
    collection_ids=[collection_id],
    limit_bm25=12,
    limit_similarity=12,
    limit_sparse=12,
    bm25_weight=0.4,
    similarity_weight=0.4,
    sparse_weight=0.2,
)

for row in rows[:5]:
    print(row.document_name, row.hybrid_score)
```

## CLI usage

### Health checks and auth

```bash
querylake --url http://127.0.0.1:8000 doctor

querylake setup \
  --url http://127.0.0.1:8000 \
  --profile local \
  --username demo \
  --password demo-pass \
  --non-interactive

querylake login \
  --url http://127.0.0.1:8000 \
  --profile local \
  --username demo \
  --password demo-pass
```

### Core RAG workflow

```bash
querylake --profile local rag create-collection --name "papers"
querylake --profile local rag list-collections
querylake --profile local rag get-collection --collection-id <id>
querylake --profile local rag update-collection --collection-id <id> --title "papers-v2"
querylake --profile local rag upload --collection-id <id> --file ./paper.pdf --await-embedding
querylake --profile local rag list-documents --collection-id <id> --limit 20
querylake --profile local rag count-chunks --collection-ids <id>
querylake --profile local rag random-chunks --collection-ids <id> --limit 5
querylake --profile local rag search --collection-id <id> --query "hybrid retrieval design" --preset tri-lane --with-metrics --min-total-results 1
```

### Directory ingest and planning

```bash
querylake --profile local rag upload-dir \
  --collection-id <id> \
  --dir ./docs \
  --pattern "*" \
  --recursive \
  --extensions ".pdf,.md" \
  --exclude-glob "archive/*" \
  --dry-run \
  --list-files \
  --selection-output ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_dry_run.json

querylake --profile local rag upload-dir \
  --collection-id <id> \
  --from-selection ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_run.json \
  --checkpoint-file ./artifacts/upload_checkpoint.json

querylake --profile local rag upload-dir \
  --collection-id <id> \
  --from-selection ./artifacts/selected_files.json \
  --resume \
  --checkpoint-file ./artifacts/upload_checkpoint.json \
  --checkpoint-save-every 10 \
  --dedupe-content-hash \
  --dedupe-scope all \
  --idempotency-strategy content-hash \
  --idempotency-prefix qlsdk
```

### Lexical-only and batch retrieval flows

```bash
querylake --profile local rag search --mode bm25 --collection-id <id> --query "hybrid retrieval design"

querylake --profile local rag search-batch \
  --collection-id <id> \
  --queries-file ./queries.txt \
  --preset tri-lane \
  --with-metrics \
  --min-total-results 1 \
  --fail-on-empty \
  --output-file ./artifacts/query_batch.json
```

### Destructive cleanup

```bash
querylake --profile local rag delete-document --document-id <doc_hash_id> --yes
```

## Python usage

### Typed option helpers

```python
from querylake_sdk import HybridSearchOptions, QueryLakeClient, UploadDirectoryOptions

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

collection = client.create_collection(name="typed-demo")
collection_id = collection["hash_id"]

run = client.upload_directory_with_options(
    collection_hash_id=collection_id,
    options=UploadDirectoryOptions(
        directory="./docs",
        pattern="*.pdf",
        recursive=True,
        dedupe_by_content_hash=True,
        dedupe_scope="all",
        idempotency_strategy="content-hash",
    ),
)
print(run["uploaded"], run["failed"])

results = client.search_hybrid_with_options(
    query="What is the main contribution?",
    collection_ids=[collection_id],
    options=HybridSearchOptions(
        limit_bm25=12,
        limit_similarity=12,
        limit_sparse=12,
        bm25_weight=0.4,
        similarity_weight=0.4,
        sparse_weight=0.2,
    ),
)
print(len(results))
```

### Metrics-aware hybrid retrieval

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

payload = client.search_hybrid_with_metrics(
    query="hybrid retrieval design",
    collection_ids=["<collection-id>"],
)

print(payload.get("duration", {}))
print(payload.get("rows", [])[:3])
```

### Raw escape hatch for lower-level platform calls

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

payload = client.api(
    "search_hybrid",
    {
        "query": "vapor recovery",
        "collection_ids": ["<collection-id>"],
        "limit_bm25": 12,
        "limit_similarity": 12,
        "limit_sparse": 12,
    },
)
print(payload)
```

## Offline examples

The SDK ships with deterministic offline examples so you can inspect CLI/Python workflows without needing a live backend.

### Bulk ingest + search tutorial

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --offline-demo \
  --dir ./docs \
  --pattern "*.md" \
  --recursive \
  --query "hybrid retrieval"
```

### Batch benchmark tutorial

```bash
python examples/sdk/rag_search_batch_benchmark.py \
  --offline-demo \
  --queries-file ./examples/sdk/fixtures/offline_queries.txt \
  --output-file ./artifacts/benchmark_offline.json
```

### What the offline fixtures are good for

| Scenario | Why use offline mode |
|---|---|
| CLI/tutorial documentation | deterministic output without standing up a backend |
| SDK examples in CI or docs work | removes auth/runtime dependencies |
| local experimentation with selection/planning flows | verifies usage shape before involving a server |

## Profile management

SDK CLI profiles are stored at `~/.querylake/sdk_profiles.json`.

```bash
querylake profile list
querylake profile show --name local
querylake profile set-url --name local --url http://127.0.0.1:8000
querylake profile set-default --name local
querylake profile delete --name old-profile
```

Practical behavior:

- after `login`, the profile becomes the active default,
- so most subsequent commands can omit `--profile`,
- and you can keep separate profiles for local, staging, and hosted environments.

## Async client

```python
import asyncio
from querylake_sdk import AsyncQueryLakeClient

async def run():
    async with AsyncQueryLakeClient(base_url="http://127.0.0.1:8000") as client:
        await client.login(username="demo", password="demo-pass")
        print(await client.healthz())

asyncio.run(run())
```

## Compatibility and scope

### What the SDK promises

| Contract | Current stance |
|---|---|
| Python support | 🟢 `>=3.10` |
| QueryLake auth + collection workflows | 🟢 first-class |
| QueryLake hybrid retrieval flows | 🟢 first-class |
| OpenAI-compatible endpoint access | 🟢 supported alongside platform API usage |
| Stable developer-facing integration surface | 🟢 intended contract |

### What the SDK does not try to be

- It is not a full replacement for backend internals.
- It is not the place where retrieval semantics are defined.
- It is not a promise that every deployment profile supports every advanced backend feature.

The SDK aims to be the **cleanest way to use QueryLake**, not the place where QueryLake’s runtime architecture is reinvented.

## Documentation map

| Topic | Link | Why you would read it |
|---|---|---|
| Quickstart | [`docs/sdk/SDK_QUICKSTART.md`](../../docs/sdk/SDK_QUICKSTART.md) | practical first use |
| RAG research workflows | [`docs/sdk/RAG_RESEARCH_PLAYBOOK.md`](../../docs/sdk/RAG_RESEARCH_PLAYBOOK.md) | retrieval/eval-centric usage |
| Bulk ingest reference | [`docs/sdk/BULK_INGEST_REFERENCE.md`](../../docs/sdk/BULK_INGEST_REFERENCE.md) | dry-run, checkpoint, resume, dedupe semantics |
| API reference | [`docs/sdk/API_REFERENCE.md`](../../docs/sdk/API_REFERENCE.md) | method-level reference |
| CI profiles | [`docs/sdk/CI_PROFILES.md`](../../docs/sdk/CI_PROFILES.md) | package validation and release expectations |
| PyPI release runbook | [`docs/sdk/PYPI_RELEASE.md`](../../docs/sdk/PYPI_RELEASE.md) | publishing workflow |
| TestPyPI dry-run | [`docs/sdk/TESTPYPI_DRYRUN.md`](../../docs/sdk/TESTPYPI_DRYRUN.md) | release rehearsal |
| Live staging integration | [`docs/sdk/LIVE_STAGING_INTEGRATION.md`](../../docs/sdk/LIVE_STAGING_INTEGRATION.md) | integration environment contract |

## Development and release

### Local development commands

```bash
cd sdk/python
pip install -e ".[dev]"
pytest -q
```

From repo root, the common SDK commands are:

```bash
make sdk-install-dev
make sdk-precommit-install
make sdk-precommit-run
make sdk-lint
make sdk-type
make sdk-test
make sdk-build
make sdk-ci
make sdk-smoke
```

### Packaging lanes and release posture

| Package | Source of truth |
|---|---|
| `querylake-sdk` | `sdk/python/pyproject.toml` |
| backend/runtime package | root `pyproject.toml` |

### CI/release surfaces

| Workflow | Role |
|---|---|
| [`sdk_checks.yml`](../../.github/workflows/sdk_checks.yml) | tests, lint, type, release guard |
| [`sdk_live_integration.yml`](../../.github/workflows/sdk_live_integration.yml) | live staging contract |
| [`sdk_publish_dryrun.yml`](../../.github/workflows/sdk_publish_dryrun.yml) | scheduled/manual TestPyPI rehearsal |
| [`sdk_publish.yml`](../../.github/workflows/sdk_publish.yml) | official publish workflow |

If you are changing the SDK, treat it as an external contract. Additive changes are easy. Accidental breaking changes are expensive.
