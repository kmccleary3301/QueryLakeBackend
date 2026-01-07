from __future__ import annotations

from typing import Any, Tuple

from QueryLake.api.auth_utils import resolve_bearer_auth_header
from QueryLake.api.single_user_auth import get_user, process_input_as_auth_type
from QueryLake.typing.config import AuthInputType
from QueryLake.runtime.auth_provider import AuthProvider


class LocalAuthProvider(AuthProvider):
    """Adapter around the current QueryLake auth logic."""

    def __init__(self, database):
        self._db = database

    def issue_token(self, principal: dict, scopes: list[str]) -> str:
        raise NotImplementedError("Token issuance remains in the legacy auth flow.")

    def validate_token(self, token: str) -> dict:
        auth = resolve_bearer_auth_header(f"Bearer {token}")
        user_tuple = get_user(self._db, auth, return_auth_type=True)
        return {
            "user": user_tuple[0],
            "auth": user_tuple[1],
            "auth_type": user_tuple[3],
        }

    def refresh_token(self, token: str) -> str:
        raise NotImplementedError("Token refresh remains in the legacy auth flow.")

    def validate_auth(self, auth: AuthInputType) -> Tuple[Any, Any]:
        auth = process_input_as_auth_type(auth)
        return get_user(self._db, auth)

    def validate_auth_with_type(self, auth: AuthInputType) -> Tuple[Any, Any, Any, int]:
        auth = process_input_as_auth_type(auth)
        return get_user(self._db, auth, return_auth_type=True)
