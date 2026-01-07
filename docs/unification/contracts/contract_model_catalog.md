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
- `capabilities` (object): tool‑use, function‑calling, vision input, json‑mode, streaming
- `routing` (object): local vs passthrough, upstream id mapping, fallback order
- `status` (string): `active` | `disabled` | `deprecated`
- `limits` (object, optional): `max_tokens`, `max_input_tokens`, `max_output_tokens`
- `tags` (list, optional): `preferred`, `experimental`, `gpu_heavy`

## Queries
- `GET /v1/models` returns the catalog
- `GET /v1/models/{id}` returns a model record

## Routing Rules
- If `provider == local`, route to internal deployments by `id`.
- If `provider != local`, treat `id` as `provider/model` and use provider registry.
- `routing.upstream_id` overrides the model id sent upstream (if present).
- `routing.fallback` is an ordered list of ids for retry on upstream failure.

## Capability Flags (recommended)
- `tool_use`: supports tools/functions
- `vision`: accepts images
- `json_mode`: supports structured JSON output
- `streaming`: supports SSE streaming

## Inventory (QueryLake default_config.json)
### Local LLM / Vision Models
- `qwen2-vl-7b-instruct` (vision)
- `qwen2.5-vl-3b-instruct` (vision)
- `qwen2.5-vl-7b-instruct` (vision)
- `llava-v1.6-7b` (vision)
- `ovis2-1b` (vision)
- `ovis2-2b` (vision)
- `ovis2-4b` (vision)
- `ovis2-8b` (vision)
- `ovis2-16b` (vision)
- `molmo-7b-d` (vision)
- `llama-3.1-8b-instruct` (text)
- `llama-3.1-70b-instruct` (text)

### Embedding / Rerank / OCR
- `bge-m3` (embedding)
- `bge-reranker-v2-m3` (rerank)
- `marker` (ocr / document processing)

### External Providers (configured)
- `openai/gpt-4-1106-preview`
- `openai/gpt-3.5-turbo-1106`

_Note: BreadBoard/Hermes inventories should be appended once repo scans are available; this section is the authoritative QueryLake base list for unification._

## Compatibility
- Records must be stable and backwards compatible; new fields optional.
- Plugins must treat unknown fields as opaque.
