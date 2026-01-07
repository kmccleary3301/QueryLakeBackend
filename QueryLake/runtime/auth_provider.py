from __future__ import annotations

from typing import Dict


class AuthProvider:
    def issue_token(self, principal: dict, scopes: list[str]) -> str:
        raise NotImplementedError

    def validate_token(self, token: str) -> dict:
        raise NotImplementedError

    def refresh_token(self, token: str) -> str:
        raise NotImplementedError


AUTH_PROVIDERS: Dict[str, AuthProvider] = {}


def register_provider(name: str, provider: AuthProvider) -> None:
    AUTH_PROVIDERS[name] = provider


def get_provider(name: str) -> AuthProvider:
    return AUTH_PROVIDERS[name]
