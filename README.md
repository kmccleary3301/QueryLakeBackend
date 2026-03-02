# QueryLake Backend

A production-oriented backend for **self-hosted AI + RAG systems** with:

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- Function-style platform API (`/api/<function_name>`) for auth, collections, ingestion, toolchains, search
- Hybrid retrieval stack (BM25 + dense + sparse lanes)
- Toolchains runtime (graph execution + interface schemas)
- Ray Serve orchestration with GPU/VRAM-aware scheduling

## Why this repo exists

This repo is the backend control plane and runtime for QueryLake.
It is designed to be:

- **Research-friendly**: rapid retrieval primitive iteration and evaluation hooks
- **Developer-friendly**: stable API routes + dedicated SDK + CLI workflow
- **Ops-aware**: health probes, rollout gates, and retrieval performance instrumentation

## Repository map

| Path | Purpose |
|---|---|
| `server.py` | FastAPI + Ray Serve entrypoint |
| `QueryLake/api/` | Core platform function APIs |
| `QueryLake/runtime/` | Retrieval/runtime orchestration primitives |
| `scripts/` | Eval, stress, rollout, and ops tooling |
| `sdk/python/` | Standalone `querylake-sdk` Python package (PyPI target) |
| `docs/` | Architecture, setup, SDK, release docs |

## 5-minute local quickstart (API-only)

API-only mode is the fastest setup for developers and RAG researchers.

### 1) Bootstrap environment

```bash
cp .env.example .env
make bootstrap
```

### 2) Start Postgres/ParadeDB

```bash
make up-db
```

### 3) Run QueryLake backend

```bash
make run-api-only
```

### 4) Verify readiness

```bash
make health
make sdk-smoke
```

## SDK-first workflow (recommended)

`querylake-sdk` is the standardized way to build apps/research pipelines against QueryLake.

### Install SDK

```bash
pip install querylake-sdk
```

### CLI flow

```bash
querylake --url http://127.0.0.1:8000 doctor
querylake login --url http://127.0.0.1:8000 --profile local --username <u> --password <p>
querylake --profile local rag create-collection --name "papers"
querylake --profile local rag list-collections
querylake --profile local rag get-collection --collection-id <id>
querylake --profile local rag update-collection --collection-id <id> --title "papers-v2"
querylake --profile local rag upload --collection-id <id> --file ./paper.pdf --await-embedding
querylake --profile local rag upload-dir --collection-id <id> --dir ./docs --pattern "*.pdf" --recursive
querylake --profile local rag upload-dir --collection-id <id> --dir ./docs --pattern "*" --recursive --extensions ".pdf,.md" --exclude-glob "archive/*" --dry-run --list-files --selection-output ./artifacts/selected_files.json --report-file ./artifacts/upload_dry_run.json
querylake --profile local rag upload-dir --collection-id <id> --from-selection ./artifacts/selected_files.json --report-file ./artifacts/upload_run.json
querylake --profile local rag list-documents --collection-id <id> --limit 20
querylake --profile local rag count-chunks --collection-ids <id>
querylake --profile local rag random-chunks --collection-ids <id> --limit 5
querylake --profile local rag search --collection-id <id> --query "main contribution" --preset tri-lane --with-metrics --min-total-results 1
querylake --profile local rag search-batch --collection-id <id> --queries-file ./queries.txt --preset tri-lane --with-metrics --min-total-results 1 --fail-on-empty --output-file ./artifacts/query_batch.json
# destructive: requires explicit confirmation
querylake --profile local rag delete-document --document-id <doc_hash_id> --yes
```

### Python flow

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

plan = client.upload_directory(
    collection_hash_id=collection_id,
    directory="./docs",
    pattern="*",
    recursive=True,
    include_extensions=[".pdf", ".md"],
    exclude_globs=["archive/*", "*.tmp"],
    dry_run=True,
)
print(plan["requested_files"])

rows = client.search_hybrid(
    query="What is the key claim?",
    collection_ids=[collection_id],
    limit_bm25=12,
    limit_similarity=12,
    limit_sparse=12,
    bm25_weight=0.4,
    similarity_weight=0.4,
    sparse_weight=0.2,
)
print(rows[:3])
```

## Retrieval and RAG capabilities

- **Three-lane hybrid retrieval**: BM25 + dense vector + sparse vector
- **Configurable lane fusion**: weighted fusion and RRF-compatible workflows
- **Constraint-aware lexical retrieval**: advanced operator handling and hard-prefilter controls
- **Strict rollout and parity tooling**: retrieval gates, delta reports, and stress harnesses
- **Ingestion controls**: per-request dense/sparse embedding toggles and dimensional safety checks

## Developer setup and docs

- Backend setup: `docs/setup/DEVELOPER_SETUP.md`
- SDK quickstart: `docs/sdk/SDK_QUICKSTART.md`
- SDK RAG research playbook: `docs/sdk/RAG_RESEARCH_PLAYBOOK.md`
- SDK API reference: `docs/sdk/API_REFERENCE.md`
- SDK PyPI release runbook: `docs/sdk/PYPI_RELEASE.md`
- SDK runnable examples: `examples/sdk/`
- Route and unification docs: `docs/unification/`
- Contributor guide: `CONTRIBUTING.md`

## Packaging and PyPI

This repository now has two packaging tracks:

1. **Backend package** (`querylake-backend`) via root `pyproject.toml`
2. **SDK package** (`querylake-sdk`) via `sdk/python/pyproject.toml`

Use the standalone SDK package for application developers and researchers. Keep backend runtime dependencies isolated to backend deployments.

Release helpers:

```bash
make sdk-precommit-install
make sdk-precommit-run
make sdk-lint
make sdk-type
make sdk-ci
make sdk-release-check
# optional publish helpers
make sdk-release-testpypi
make sdk-release-pypi
```

Local CI parity helpers:

```bash
make ci-docs
make ci-unification
make ci-retrieval-smoke
```

## Runtime notes

### API-only mode

```bash
export QUERYLAKE_API_ONLY=1
```

This disables local model deployments while preserving API/RAG/toolchain functionality.

### Health endpoints

- `GET /healthz`
- `GET /readyz`
- `GET /api/ping`

## Contributing guidelines (practical)

- Prefer additive API changes with backward compatibility.
- Keep retrieval primitive changes paired with evaluation artifacts.
- Treat SDK stability as an external contract: add methods, avoid breaking existing ones.
- Document setup/runtime changes under `docs/` in the same PR.

## License

Apache-2.0 (`LICENSE`).
