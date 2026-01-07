# Contract F — Events & Backfill

## Purpose
Define event schema and replay semantics.

## Event Record
- `event_id`, `request_id`, `run_id`, `job_id`
- `type`, `timestamp`, `payload`
- `actor_id` (optional)
- `session_id` (optional)

## Backfill
- support replay by `run_id` or `session_id`
- retention policy documented

## Replay Rules
- Events are immutable and ordered by `event_id` or `timestamp`
- Replay should be idempotent (no side effects)
- Retention default: 30 days unless configured
