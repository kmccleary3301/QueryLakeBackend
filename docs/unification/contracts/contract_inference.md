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

## Streaming Semantics
- Each chunk is a JSON object with `choices[].delta`
- The first chunk may include role/content
- Terminal message is `data: [DONE]`

## Error Model
- `429` for rate limits
- `5xx` for upstream / internal failure
- Error body includes `object=error`, `type`, `message`

## Status Codes (recommendations)
- `400` invalid request (missing model, bad args)
- `401` auth required
- `403` forbidden (scope/permissions)
- `404` model not found
- `429` rate limit / concurrency limit
- `5xx` internal or upstream failure

## Tool / Function Call Schema
- Requests may include `tools` (OpenAI-style) or `functions` (legacy)
- Tool definition should map to `FunctionCallDefinition`:
  - `name` (string)
  - `description` (string)
  - `parameters` (list of `FunctionCallArgumentDefinition` with `name/type/default/description`)
- Responses may include `tool_calls` or `function_call` in `choices[].delta`
- Tool name resolution is case-sensitive and must match the registered function name
