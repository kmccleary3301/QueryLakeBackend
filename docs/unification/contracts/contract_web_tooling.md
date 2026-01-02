# Contract D — Web Tooling

## Purpose
Standardize search, scrape, and crawl interfaces for Hermes and QueryLake.

## Search
- Request: query, provider, optional domain filters
- Response: list of results with provenance

## Scrape
- Request: url, timeout, markdown option
- Response: artifact or inline content

## Crawl
- Request: seed urls, limits, depth
- Response: job id, status, artifacts

