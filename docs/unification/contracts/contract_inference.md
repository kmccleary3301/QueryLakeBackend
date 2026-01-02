# Contract B — Inference

## Purpose
Define the standardized LLM call interface used across QueryLake and plugins.

## Endpoints
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

## Request
- OpenAI‑compatible JSON payloads
- Required: `model`
- Optional: `messages`, `input`, `temperature`, `max_tokens`, `stream`

## Response
- OpenAI‑compatible response shape
- Streaming: SSE `data:` chunks + `[DONE]` terminator

## Error Model
- `429` for rate limits
- `5xx` for upstream / internal failure
- Error body includes `object=error`, `type`, `message`

