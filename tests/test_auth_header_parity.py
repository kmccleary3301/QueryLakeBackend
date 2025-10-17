from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.typing.config import AuthType2, AuthType4
from QueryLake.api.auth_utils import resolve_bearer_auth_header


def test_bearer_api_key_header_parsed_as_auth_type2():
    auth = resolve_bearer_auth_header("Bearer sk-abc123")
    assert isinstance(auth, AuthType2)
    assert auth.api_key.startswith("sk-")


def test_bearer_oauth_header_parsed_as_auth_type4():
    auth = resolve_bearer_auth_header("Bearer some.jwt.token")
    assert isinstance(auth, AuthType4)
    assert auth.oauth2 == "some.jwt.token"

