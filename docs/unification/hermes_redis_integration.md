# Hermes Redis Integration (Draft)

## Goals
- Reliable crawl job persistence
- Distributed queues for workers
- Support retries + backoff + failure tracking

## Data Structures
- `hermes:job:{id}` (hash): job metadata + state
- `hermes:queue:pending` (list or zset)
- `hermes:queue:retry` (zset with timestamps)

## Workflow
1. Submit crawl job → create hash + enqueue id
2. Worker claims job (atomic pop)
3. Worker updates progress + artifacts
4. On failure: requeue with backoff

## Requirements
- Redis required for Hermes integration
- QueryLake should provide shared Redis
- Redis-backed HermesQueue utility added for enqueue/dequeue/retry
- HermesQueue supports `requeue_due` for retry drainage

## Reliability Validation (next)
- Inject worker failure mid-job → verify retry schedule
- Force Redis disconnect → ensure job remains in pending/retry
- Validate idempotent enqueue for duplicate job_id
