from __future__ import annotations

from QueryLake.runtime.auth_provider import AuthProvider


class OAuthAuthProvider(AuthProvider):
    """Placeholder for OAuth/OIDC provider integration."""

    def issue_token(self, principal: dict, scopes: list[str]) -> str:
        raise NotImplementedError("OAuth provider not configured")

    def validate_token(self, token: str) -> dict:
        raise NotImplementedError("OAuth provider not configured")

    def refresh_token(self, token: str) -> str:
        raise NotImplementedError("OAuth provider not configured")
