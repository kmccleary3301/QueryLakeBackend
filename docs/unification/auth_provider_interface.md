# Auth Provider Interface (Draft)

## Purpose
Abstract auth providers to allow future OAuth/OIDC while preserving existing API key and session flows.

## Required Functions
- `issue_token(principal, scopes) -> token`
- `validate_token(token) -> principal`
- `refresh_token(token) -> token`

## Current Implementation
- API key flow in QueryLake
- Username/password for user auth

## Next Steps
- Wrap current auth in adapter class
- Add provider registry
- Add tests for provider boundary

