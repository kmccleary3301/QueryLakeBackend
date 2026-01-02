# Contract A — Model Catalog

## Purpose
Define a single, canonical model registry used by QueryLake + plugins (BreadBoard/Hermes) for discovery, routing, and capability checks.

## Model Record Schema (canonical)
- `id` (string, required): stable model identifier
- `display_name` (string, optional)
- `provider` (string, required): `local` | `openai` | `anthropic` | `custom`
- `engine` (string, optional): `vllm` | `exllamav2` | `remote`
- `modalities` (list, required): `text` | `vision` | `audio` | `embedding` | `rerank`
- `max_context` (int, optional)
- `capabilities` (object): tool‑use, function‑calling, vision input, etc.
- `routing` (object): local vs passthrough, upstream id mapping
- `status` (string): `active` | `disabled` | `deprecated`

## Queries
- `GET /v1/models` returns the catalog
- `GET /v1/models/{id}` returns a model record

## Compatibility
- Records must be stable and backwards compatible; new fields optional.
- Plugins must treat unknown fields as opaque.

