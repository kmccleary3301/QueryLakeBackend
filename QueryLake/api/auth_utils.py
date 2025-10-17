from __future__ import annotations

from QueryLake.typing.config import AuthType
from QueryLake.api.single_user_auth import process_input_as_auth_type


def resolve_bearer_auth_header(auth_header: str) -> AuthType:
    """Parse an Authorization: Bearer header value into an AuthType.

    Heuristic: API keys begin with "sk-"; otherwise treat as OAuth2 bearer.
    """
    token = auth_header.split(" ", 1)[1] if " " in auth_header else auth_header
    if token.startswith("sk-"):
        return process_input_as_auth_type({"api_key": token})
    return process_input_as_auth_type(token)

