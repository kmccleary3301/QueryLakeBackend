# Billing / Usage Accounting (Draft)

## Goals
- Private-first, billing optional
- Usage accounting consistent across providers

## Usage Event Fields
- request_id
- principal_id
- model
- input_tokens / output_tokens
- status
- provider (local/external)

## Future
- Credits / account ledger optional
- External provider passthrough usage recorded
- Usage events logged from llm/embedding/rerank calls

## Accounting Stub (v1)
- Usage events include `principal_id` and `provider` fields for later billing
- Logged via `querylake.usage` logger for ingestion
