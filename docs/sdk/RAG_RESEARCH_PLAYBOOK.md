# QueryLake RAG Research Playbook

This guide is a practical reference for running repeatable RAG experiments with QueryLake.

It is designed for:
- researchers iterating on retrieval mixes and constraints
- developers validating ingestion/search behavior quickly

## 1) Baseline setup

```bash
querylake --url http://127.0.0.1:8000 doctor
querylake login --url http://127.0.0.1:8000 --profile local --username <username> --password <password>
```

Create a collection:

```bash
querylake --profile local rag create-collection --name "rag-playbook"
querylake --profile local rag list-collections
querylake --profile local rag get-collection --collection-id <collection_id>
querylake --profile local rag update-collection --collection-id <collection_id> --title "rag-playbook-v2"
```

## 2) Ingestion profiles

### Dry-run planning (select files before upload)

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*" \
  --recursive \
  --extensions ".txt,.md" \
  --exclude-glob "archive/*" \
  --exclude-glob "*.tmp" \
  --dry-run \
  --list-files \
  --selection-output ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_dry_run.json
```

### Dense-only (cheapest baseline)

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive
```

### Dense + sparse (3-lane-ready)

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive \
  --sparse-embeddings
```

### Full blocking ingest (small corpora / deterministic runs)

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive \
  --await-embedding
```

## 3) Retrieval profiles

### Hybrid default (BM25 + dense)

```bash
querylake --profile local rag search \
  --collection-id <collection_id> \
  --query "main contribution" \
  --preset balanced \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --limit-sparse 0 \
  --bm25-weight 0.55 \
  --similarity-weight 0.45 \
  --min-total-results 1 \
  --with-metrics

# Inspect indexed state for debugging
querylake --profile local rag list-documents --collection-id <collection_id> --limit 20
querylake --profile local rag count-chunks --collection-ids <collection_id>
querylake --profile local rag random-chunks --collection-ids <collection_id> --limit 5
# Cleanup one noisy document (destructive)
querylake --profile local rag delete-document --document-id <document_hash_id> --yes
```

### Three-lane hybrid (BM25 + dense + sparse)

```bash
querylake --profile local rag search \
  --collection-id <collection_id> \
  --query "main contribution" \
  --preset tri-lane \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --limit-sparse 12 \
  --bm25-weight 0.40 \
  --similarity-weight 0.40 \
  --sparse-weight 0.20
```

### Lexical-only constraint check (BM25 control path)

```bash
querylake --profile local rag search \
  --mode bm25 \
  --collection-id <collection_id> \
  --query "\"exact phrase\" +must -exclude site:docs.example.com"

# Batch execution from file (one query per line)
querylake --profile local rag search-batch \
  --collection-id <collection_id> \
  --queries-file ./queries.txt \
  --preset tri-lane \
  --min-total-results 1 \
  --fail-on-empty \
  --with-metrics \
  --output-file ./artifacts/query_batch.json
```

## 4) Python experiment harness

Use the runnable script:

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --base-url http://127.0.0.1:8000 \
  --username <username> \
  --password <password> \
  --collection rag-playbook-python \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive \
  --query "main contribution" \
  --sparse-embeddings \
  --limit-bm25 12 \
  --limit-similarity 12 \
  --limit-sparse 12 \
  --bm25-weight 0.4 \
  --similarity-weight 0.4 \
  --sparse-weight 0.2
```

## 5) Practical tuning notes

- Start with dense-only for fast smoke tests.
- Enable sparse lane once your ingestion throughput/cost budget is validated.
- Keep lane limits balanced before tuning weights; skewed limits can hide useful signals.
- Use lexical-only mode to validate advanced operator behavior separately from semantic lanes.
- Track evaluation deltas with existing retrieval CI harnesses before promoting profile changes.

## 6) Suggested experiment loop

1. Ingest with a fixed profile and record ingest settings.
2. Run baseline retrieval with stable query set.
3. Sweep one variable at a time:
   - lane limits (`limit_bm25`, `limit_similarity`, `limit_sparse`)
   - lane weights (`bm25_weight`, `similarity_weight`, `sparse_weight`)
4. Keep best candidate and verify with parity/stress checks.
5. Only then roll into strict preset workflows.
