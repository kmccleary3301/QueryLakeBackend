# Billing / Usage Accounting (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Usage-accounting and optional billing notes for keeping QueryLake private-first while preserving future ledger/billing hooks.

| Field | Value |
|---|---|
| Audience | Backend/platform maintainers, operators, future billing/usage owners |
| Use this when | Use this when you need to understand what request-level usage data should be captured now so billing can remain optional later. |
| Prerequisites | Basic familiarity with QueryLake inference/retrieval request paths and logging. |
| Related docs | [`observability_v1.md`](observability_v1.md), [`program_control.md`](program_control.md) |
| Status | 🔵 draft future-proofing note |

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
