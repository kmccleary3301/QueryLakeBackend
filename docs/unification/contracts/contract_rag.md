# Contract C — RAG Ingestion & Retrieval

## Purpose
Define ingestion inputs, retrieval queries, and evidence return formats.

## Ingestion
- Input: artifact references (files, URLs, or buffers)
- Output: ingestion job id + status lifecycle

## Retrieval
- Input: query, filters (metadata, tags), top_k
- Output: list of results with citations

## Evidence Format
- `source_id`, `source_url`, `content_hash`, `timestamp`

