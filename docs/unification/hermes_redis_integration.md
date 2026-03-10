# Hermes Redis Integration (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Redis-backed Hermes queue and retry model notes for the QueryLake/Hermes integration boundary.

| Field | Value |
|---|---|
| Audience | Hermes maintainers, infra operators, backend integrators |
| Use this when | Use this when you are validating Hermes job persistence, retry behavior, or shared Redis deployment expectations. |
| Prerequisites | Basic familiarity with Hermes crawl jobs and QueryLake Redis-backed components. |
| Related docs | [`observability_v1.md`](observability_v1.md), [`node_cloud_plan.md`](node_cloud_plan.md), [`program_control.md`](program_control.md) |
| Status | 🔵 draft integration note |

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
