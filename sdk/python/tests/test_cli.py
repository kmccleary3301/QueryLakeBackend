from __future__ import annotations

import json

import querylake_sdk.cli as cli


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "http://localhost")

    def close(self):
        return None

    def login(self, *, username: str, password: str):
        assert username == "alice"
        assert password == "secret"
        return {"auth": "tok_demo"}

    def healthz(self):
        return {"ok": True}

    def readyz(self):
        return {"ok": True}

    def ping(self):
        return {"pong": True}

    def list_models(self):
        return {"data": []}


def test_cli_login_saves_profile(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    code = cli.main(
        [
            "login",
            "--url",
            "http://127.0.0.1:8000",
            "--profile",
            "local",
            "--username",
            "alice",
            "--password",
            "secret",
        ]
    )
    assert code == 0

    payload = json.loads((home / ".querylake" / "sdk_profiles.json").read_text(encoding="utf-8"))
    assert payload["profiles"]["local"]["auth"]["oauth2"] == "tok_demo"
    captured = capsys.readouterr()
    assert "saved_profile" in captured.out


def test_cli_doctor(monkeypatch, capsys):
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)
    code = cli.main(["--url", "http://127.0.0.1:8000", "doctor"])
    assert code == 0
    captured = capsys.readouterr()
    assert "\"ok\": true" in captured.out.lower()
