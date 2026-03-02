# QueryLake Python SDK

Official lightweight Python SDK for QueryLake.

`querylake-sdk` gives you:
- Sync and async clients.
- Typed RAG search result helpers.
- Built-in CLI (`querylake`) for login profiles, health checks, and quick RAG workflows.
- Full compatibility with QueryLake `/api/*` function routes and OpenAI-compatible `/v1/*` routes.

---

## Install

```bash
pip install querylake-sdk
```

For local development from this repository:

```bash
cd sdk/python
pip install -e ".[dev]"
```

---

## Quickstart (Python)

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

results = client.search_hybrid_chunks(
    query="What is the main contribution?",
    collection_ids=[collection_id],
    limit_bm25=12,
    limit_similarity=12,
    limit_sparse=12,
    bm25_weight=0.4,
    similarity_weight=0.4,
    sparse_weight=0.2,
)

for row in results[:5]:
    print(row.document_name, row.hybrid_score)

# Need stage timings/metrics?
metrics_payload = client.search_hybrid_with_metrics(
    query="What is the main contribution?",
    collection_ids=[collection_id],
)
print(metrics_payload.get("duration", {}))
```

---

## CLI

### 1) Health checks

```bash
querylake --url http://127.0.0.1:8000 doctor
```

### 2) Login + store profile

```bash
querylake login \
  --url http://127.0.0.1:8000 \
  --profile local \
  --username demo \
  --password demo-pass
```

### 3) Create collection, upload, search

```bash
querylake --profile local rag create-collection --name "papers"
querylake --profile local rag list-collections
querylake --profile local rag upload --collection-id <id> --file ./paper.pdf --await-embedding
querylake --profile local rag upload-dir --collection-id <id> --dir ./docs --pattern "*.pdf" --recursive
querylake --profile local rag list-documents --collection-id <id> --limit 20
querylake --profile local rag count-chunks --collection-ids <id>
querylake --profile local rag search --collection-id <id> --query "hybrid retrieval design" --with-metrics
# lexical-only control path (direct BM25)
querylake --profile local rag search --mode bm25 --collection-id <id> --query "hybrid retrieval design"
```

Profiles are stored at `~/.querylake/sdk_profiles.json`.
The login profile becomes the active default, so you can omit `--profile` afterward.

### Runnable bulk example

From repo root:

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --base-url http://127.0.0.1:8000 \
  --username demo \
  --password demo-pass \
  --collection sdk-bulk-demo \
  --dir ./docs \
  --pattern "*.md" \
  --recursive \
  --query "hybrid retrieval"
```

### Profile management

```bash
querylake profile list
querylake profile show --name local
querylake profile set-url --name local --url http://127.0.0.1:8000
querylake profile set-default --name local
querylake profile delete --name old-profile
```

---

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

---

## SDK scope

- Designed for fast iteration in research and developer workflows.
- Keeps auth/session and low-level route details out of your app code.
- Leaves advanced experimentation flexible via raw `client.api("function_name", payload)` calls.

See also:
- `docs/sdk/SDK_QUICKSTART.md`
- `docs/sdk/RAG_RESEARCH_PLAYBOOK.md`
- `docs/sdk/API_REFERENCE.md`
