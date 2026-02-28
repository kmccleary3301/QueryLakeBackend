# QueryLake SDK Quickstart

This guide is the fastest way to go from zero to a working QueryLake RAG workflow.

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

## 4) Create a collection and upload a document

```bash
querylake --profile local rag create-collection --name "quickstart"
querylake --profile local rag upload --collection-id <collection_id> --file ./paper.pdf --await-embedding
```

## 5) Hybrid search

```bash
querylake --profile local rag search \
  --collection-id <collection_id> \
  --query "What is the main claim?" \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --limit-sparse 12 \
  --bm25-weight 0.4 \
  --similarity-weight 0.4 \
  --sparse-weight 0.2

# Optional lexical-only control path
querylake --profile local rag search \
  --mode bm25 \
  --collection-id <collection_id> \
  --query "main claim"
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
