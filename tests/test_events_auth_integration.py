from pathlib import Path
import sys
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api.auth_utils import resolve_bearer_auth_header


def test_events_route_accepts_bearer_api_key_and_oauth():
    app = FastAPI()

    @app.get("/sessions/{session_id}/events")
    async def list_events(session_id: str, request: Request):
        hdr = request.headers.get("authorization")
        if hdr and hdr.lower().startswith("bearer "):
            auth = resolve_bearer_auth_header(hdr)
            # Return the detected auth class name for assertion
            return {"ok": True, "auth_kind": auth.__class__.__name__}
        return {"ok": False}

    client = TestClient(app)
    r1 = client.get("/sessions/s1/events", headers={"Authorization": "Bearer sk-abc123"})
    assert r1.status_code == 200
    assert r1.json()["auth_kind"].endswith("AuthType2")

    r2 = client.get("/sessions/s1/events", headers={"Authorization": "Bearer some.jwt.token"})
    assert r2.status_code == 200
    assert r2.json()["auth_kind"].endswith("AuthType4")

