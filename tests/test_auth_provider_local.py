from QueryLake.runtime.auth_provider_local import LocalAuthProvider


def test_local_auth_provider_validate(monkeypatch):
    class DummyDB:
        pass

    def _fake_get_user(db, auth, return_auth_type=False):
        return ("user", "auth", "orig", 2)

    monkeypatch.setattr("QueryLake.runtime.auth_provider_local.get_user", _fake_get_user)

    provider = LocalAuthProvider(DummyDB())
    result = provider.validate_token("sk-test")
    assert result["auth_type"] == 2
