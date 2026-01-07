# Contract C — RAG Ingestion & Retrieval

## Purpose
Define ingestion inputs, retrieval queries, and evidence return formats.

## Ingestion
- Input: artifact references (files, URLs, or buffers)
- Output: ingestion job id + status lifecycle

### Ingestion Request (suggested)
- `artifacts[]`: `{type: file|url|buffer, id?, url?, bytes_cas?}`
- `metadata`: `{collection_id?, tags?, source?}`
- `chunking`: `{strategy?, max_tokens?, overlap?}`
- `options`: `{dedup?, language?}`

## Retrieval
- Input: query, filters (metadata, tags), top_k
- Output: list of results with citations

### Retrieval Request
- `query` (string)
- `top_k` (int, default 5)
- `filters`: `{collection_id?, tags?, time_range?}`
- `rerank`: `{model?, enabled?}`

### Retrieval Response
- `results[]`: `{text, score, citation}`
- `citation`: `{source_id, source_url?, chunk_id?, content_hash, timestamp}`

## Evidence Format
- `source_id`, `source_url`, `content_hash`, `timestamp`
- `chunk_id`, `offset_start`, `offset_end` (optional)
