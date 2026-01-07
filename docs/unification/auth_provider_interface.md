# Auth Provider Interface (Draft)

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
