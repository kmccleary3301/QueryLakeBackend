from __future__ import annotations

import json

import querylake_sdk.cli as cli


class _FakeClient:
    last_base_url = None

    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "http://localhost")
        _FakeClient.last_base_url = self.base_url

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


def test_cli_profile_default_url_resolution(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home2"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    # Save a profile and make it active.
    cli.main(
        [
            "login",
            "--url",
            "http://example.local:8001",
            "--profile",
            "work",
            "--username",
            "alice",
            "--password",
            "secret",
        ]
    )

    code = cli.main(["doctor"])
    assert code == 0
    assert _FakeClient.last_base_url == "http://example.local:8001"
    _ = capsys.readouterr()


def test_cli_profile_set_url_show_delete(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home3"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    cli.main(
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

    assert cli.main(["profile", "set-url", "--name", "local", "--url", "http://new.url:9000"]) == 0
    out = capsys.readouterr().out
    assert "new.url:9000" in out

    assert cli.main(["profile", "show", "--name", "local"]) == 0
    out = capsys.readouterr().out
    assert "\"profile\": \"local\"" in out

    assert cli.main(["profile", "delete", "--name", "local"]) == 0
    out = capsys.readouterr().out
    assert "\"deleted_profile\": \"local\"" in out
