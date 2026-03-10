# Auth Provider Interface (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Auth-provider abstraction notes for keeping local auth stable while future OAuth/OIDC support is introduced behind a registry boundary.

| Field | Value |
|---|---|
| Audience | Backend/auth maintainers, security reviewers, platform integrators |
| Use this when | Use this when you are touching token issuance/validation flows or planning external identity-provider support. |
| Prerequisites | Working knowledge of current local auth flows and QueryLake request authentication. |
| Related docs | [`api_strategy.md`](api_strategy.md), [`program_control.md`](program_control.md), [`../setup/DEVELOPER_SETUP.md`](../setup/DEVELOPER_SETUP.md) |
| Status | 🔵 draft interface note |

## Purpose
Abstract auth providers to allow future OAuth/OIDC while preserving existing API key and session flows.

## Required Functions
- `issue_token(principal, scopes) -> token`
- `validate_token(token) -> principal`
- `refresh_token(token) -> token`

## Current Implementation
- API key flow in QueryLake (LocalAuthProvider adapter)
- Username/password for user auth (LocalAuthProvider)

## Next Steps
- Add provider registry (complete)
- Add tests for provider boundary (complete)
- Stub external providers (next: OAuth/OIDC placeholder)

## OAuth/OIDC Stub
- OAuthAuthProvider added as a placeholder implementation
- Not registered by default; enable with `QL_AUTH_OAUTH_ENABLED=1`

## Registry Skeleton (planned)
```python
class AuthProvider:
    def issue_token(self, principal, scopes): ...
    def validate_token(self, token): ...
    def refresh_token(self, token): ...

AUTH_PROVIDERS = {"local": LocalAuthProvider()}
```
