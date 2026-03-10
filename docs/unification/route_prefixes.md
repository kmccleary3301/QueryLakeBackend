# Route Prefixes (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Route-prefix naming notes for separating public compatibility surfaces from internal kernel and plugin routes.

| Field | Value |
|---|---|
| Audience | Backend maintainers, API designers, SDK/runtime integrators |
| Use this when | Use this when introducing or documenting route families and deciding whether a path belongs in `/v1`, `/v2/kernel`, or `/v2/plugins`. |
| Prerequisites | Familiarity with QueryLake route organization and the API strategy workstream. |
| Related docs | [`api_strategy.md`](api_strategy.md), [`../sdk/API_REFERENCE.md`](../sdk/API_REFERENCE.md) |
| Status | 🔵 draft route note |

## Public vs Internal
- `/v1/*` — public compatibility endpoints
- `/v2/kernel/*` — internal kernel operations
- `/v2/plugins/*` — plugin surfaces

## Migration
- Maintain legacy routes for backward compatibility
- Add explicit documentation for new prefixes

## Mapping (initial)
- Legacy `/api/*` → `/v2/kernel/*` where applicable
- Legacy `/files/*` → `/v2/kernel/files/*`
- Legacy `/sessions/*` → `/v2/kernel/sessions/*`
- Legacy `/toolchains/*` → `/v2/plugins/toolchains/*`
