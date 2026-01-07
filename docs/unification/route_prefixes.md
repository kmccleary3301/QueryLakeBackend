# Route Prefixes (Draft)

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
