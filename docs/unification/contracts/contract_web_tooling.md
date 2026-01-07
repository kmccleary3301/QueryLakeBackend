# Contract D — Web Tooling

## Purpose
Standardize search, scrape, and crawl interfaces for Hermes and QueryLake.

## Search
- Request: query, provider, optional domain filters
- Response: list of results with provenance

### Search Request (example fields)
- `query` (string)
- `provider` (string, optional)
- `domains` (list, optional)
- `recency_days` (int, optional)

### Search Response
- `results[]`: `{title, url, snippet, score?, timestamp?}`

## Scrape
- Request: url, timeout, markdown option
- Response: artifact or inline content

### Scrape Response
- `content` (string) or `artifact_id`
- `metadata`: `{content_type, fetched_at, source_url}`

## Crawl
- Request: seed urls, limits, depth
- Response: job id, status, artifacts

### Crawl Job Model
- `job_id`, `status` (`pending|running|failed|completed`)
- `stats`: pages discovered, pages fetched, failures
- `artifacts`: list of output artifacts (per page)

## Evidence Record
- `source_url`
- `fetched_at` timestamp
- `content_hash` (sha256)
