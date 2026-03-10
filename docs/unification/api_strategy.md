# API Strategy (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Current route and compatibility strategy for preserving stable ingress while the internal kernel surface evolves.

| Field | Value |
|---|---|
| Audience | Backend maintainers, API surface owners, SDK/runtime integrators |
| Use this when | Use this when you are changing public routes, adding v2 kernel/plugin prefixes, or deciding whether a legacy endpoint should freeze or move. |
| Prerequisites | Familiarity with current QueryLake HTTP routes and the repo-level unification effort. |
| Related docs | [`route_prefixes.md`](route_prefixes.md), [`repo_migration.md`](repo_migration.md), [`../sdk/API_REFERENCE.md`](../sdk/API_REFERENCE.md) |
| Status | 🔵 draft strategy note |

## Decision
- Freeze legacy endpoints where practical.
- Provide compatibility shims for new ingress routes.

## Actions
- Identify legacy endpoints to freeze.
- Map legacy endpoints to new stable routes.
- Add explicit routing documentation.

## Proposed Stable Routes (v1 freeze)
- `/v1/chat/completions` (OpenAI compatible)
- `/v1/embeddings` (OpenAI compatible)
- `/api/ping`
- `/api/llm`, `/api/embedding`, `/api/rerank`
- `/files/*` (file store + events + jobs)
- `/sessions/*` (toolchain sessions)
- `/upload_document` + `/update_documents`

## New Prefix Targets (v2)
- `/v2/kernel/*` for internal orchestration (sessions, jobs, events)
- `/v2/plugins/*` for toolchain-facing plugins
- `/v2/admin/*` for admin-only operations

## Compatibility Plan
- Keep legacy endpoints indefinitely (freeze behavior).
- Add new v2 endpoints as opt-in (no behavior changes).
- Document every legacy endpoint with v2 counterpart mapping.

## Implemented Shims (v2 kernel)
- `/v2/kernel/chat/completions` → `/v1/chat/completions`
- `/v2/kernel/embeddings` → `/v1/embeddings`
- `/v2/kernel/files/*` → `/files/*`
- `/v2/kernel/sessions/*` → `/sessions/*`
- `/v2/kernel/upload_document` + `/v2/kernel/update_documents` → legacy handlers
