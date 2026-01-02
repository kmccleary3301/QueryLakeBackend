# Contract F — Events & Backfill

## Purpose
Define event schema and replay semantics.

## Event Record
- `event_id`, `request_id`, `run_id`, `job_id`
- `type`, `timestamp`, `payload`

## Backfill
- support replay by `run_id` or `session_id`
- retention policy documented

