# QueryLake SDK Quickstart

This guide is the fastest way to go from zero to a working QueryLake RAG workflow.

For contributors validating packaging quality locally, run:

```bash
make sdk-precommit-install
make sdk-precommit-run
make sdk-lint
make sdk-type
make sdk-ci
```

## 1) Install SDK

```bash
pip install querylake-sdk
```

## 2) Verify backend health

```bash
querylake --url http://127.0.0.1:8000 doctor
```

## 3) Login and store profile

```bash
querylake login \
  --url http://127.0.0.1:8000 \
  --profile local \
  --username <username> \
  --password <password>
```

`login` also sets this profile as the active default profile.

## 4) Create a collection and upload a document

```bash
querylake rag create-collection --name "quickstart"
querylake rag list-collections
querylake rag get-collection --collection-id <collection_id>
querylake rag update-collection --collection-id <collection_id> --title "quickstart-v2"
querylake rag upload --collection-id <collection_id> --file ./paper.pdf --await-embedding

# Bulk upload a folder (helpful for experiments)
querylake rag upload-dir \
  --collection-id <collection_id> \
  --dir ./docs \
  --pattern "*.pdf" \
  --recursive

# Apply a named ingest baseline profile for repeatable runs
querylake rag upload-dir \
  --collection-id <collection_id> \
  --dir ./docs \
  --pattern "*.txt" \
  --recursive \
  --ingest-profile tri-lane-fast

# Preview selection only (no upload), with include/exclude filters
querylake rag upload-dir \
  --collection-id <collection_id> \
  --dir ./docs \
  --pattern "*" \
  --recursive \
  --extensions ".pdf,.md" \
  --exclude-glob "archive/*" \
  --dry-run \
  --list-files \
  --selection-output ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_dry_run.json

# Re-run upload from a saved selection file (exact same file set)
querylake rag upload-dir \
  --collection-id <collection_id> \
  --from-selection ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_run.json \
  --checkpoint-file ./artifacts/upload_checkpoint.json

# Resume interrupted upload from checkpoint
querylake rag upload-dir \
  --collection-id <collection_id> \
  --from-selection ./artifacts/selected_files.json \
  --resume \
  --checkpoint-file ./artifacts/upload_checkpoint.json \
  --checkpoint-save-every 10 \
  --dedupe-content-hash \
  --dedupe-scope all \
  --idempotency-strategy content-hash \
  --idempotency-prefix qlsdk \
  --report-file ./artifacts/upload_resume.json

# Inspect corpus state
querylake rag list-documents --collection-id <collection_id> --limit 20 --offset 0
querylake rag count-chunks --collection-ids <collection_id>
querylake rag random-chunks --collection-ids <collection_id> --limit 5

# Destructive cleanup for one document (requires explicit confirmation)
querylake rag delete-document --document-id <document_hash_id> --yes
```

## 5) Hybrid search

```bash
querylake rag search \
  --collection-id <collection_id> \
  --query "What is the main claim?" \
  --preset tri-lane \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --limit-sparse 12 \
  --bm25-weight 0.4 \
  --similarity-weight 0.4 \
  --sparse-weight 0.2 \
  --min-total-results 1 \
  --with-metrics

# Optional lexical-only control path
querylake rag search \
  --mode bm25 \
  --collection-id <collection_id> \
  --query "main claim"

# Batch query file (one query per line, '#' comments allowed)
querylake rag search-batch \
  --collection-id <collection_id> \
  --queries-file ./queries.txt \
  --preset tri-lane \
  --min-total-results 1 \
  --fail-on-empty \
  --with-metrics \
  --output-file ./artifacts/batch_results.json
```

## 6) Profile management (optional)

```bash
querylake profile list
querylake profile show
querylake profile set-url --url http://127.0.0.1:8000
querylake profile set-default --name local
querylake profile delete --name stale-profile
```

## Python API (sync)

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

collection = client.create_collection(name="quickstart")
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

run = client.upload_directory(
    collection_hash_id=collection_id,
    directory="./docs",
    pattern="*.pdf",
    recursive=True,
    checkpoint_file="./artifacts/upload_checkpoint.json",
    checkpoint_save_every=10,
    dedupe_by_content_hash=True,
    dedupe_scope="all",
    idempotency_strategy="content-hash",
    idempotency_prefix="qlsdk",
)
print(run["uploaded"], run["failed"])

rows = client.search_hybrid_chunks(
    query="main contribution",
    collection_ids=[collection_id],
    limit_bm25=12,
    limit_similarity=12,
    limit_sparse=12,
    bm25_weight=0.4,
    similarity_weight=0.4,
    sparse_weight=0.2,
)
print(rows[0].text if rows else "No match")
```

## Python API (async)

```python
import asyncio
from querylake_sdk import AsyncQueryLakeClient

async def run():
    async with AsyncQueryLakeClient(base_url="http://127.0.0.1:8000") as client:
        await client.login(username="demo", password="demo-pass")
        print(await client.healthz())

asyncio.run(run())
```

## Runnable example script

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --base-url http://127.0.0.1:8000 \
  --username <username> \
  --password <password> \
  --collection sdk-bulk-demo \
  --dir ./docs \
  --pattern "*.pdf" \
  --recursive \
  --query "main contribution"

# Plan-only pass with saved artifacts
python examples/sdk/rag_bulk_ingest_and_search.py \
  --base-url http://127.0.0.1:8000 \
  --collection sdk-bulk-demo \
  --dir ./docs \
  --pattern "*" \
  --recursive \
  --extensions ".md,.txt" \
  --exclude-glob "archive/*" \
  --dry-run \
  --selection-output ./artifacts/selected_files.json \
  --upload-report-file ./artifacts/upload_dry_run.json \
  --query "main contribution"

# Fully offline deterministic mode (no backend/auth required)
python examples/sdk/rag_bulk_ingest_and_search.py \
  --offline-demo \
  --dir ./docs \
  --pattern "*.md" \
  --recursive \
  --query "main contribution"

# Offline batch benchmark helper
python examples/sdk/rag_search_batch_benchmark.py \
  --offline-demo \
  --queries-file ./examples/sdk/fixtures/offline_queries.txt \
  --output-file ./artifacts/benchmark_offline.json
```

For deeper retrieval/ingestion tuning patterns, see `docs/sdk/RAG_RESEARCH_PLAYBOOK.md`.
