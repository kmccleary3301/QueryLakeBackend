from __future__ import annotations

import json

import querylake_sdk.cli as cli


class _FakeClient:
    last_base_url = None

    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "http://localhost")
        _FakeClient.last_base_url = self.base_url
        self.uploaded = []

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

    def list_collections(self, *, organization_id=None, global_collections=False):
        return {
            "collections": [
                {
                    "hash_id": "col_1",
                    "name": "demo",
                    "organization_id": organization_id,
                    "global": bool(global_collections),
                }
            ]
        }

    def list_collection_documents(self, *, collection_hash_id, limit=100, offset=0):
        return [
            {
                "hash_id": "doc_1",
                "collection_hash_id": str(collection_hash_id),
                "limit_used": int(limit),
                "offset_used": int(offset),
            }
        ]

    def count_chunks(self, *, collection_ids=None):
        return {
            "chunk_count": 42,
            "collection_ids": list(collection_ids) if collection_ids is not None else None,
        }

    def search_hybrid(self, **kwargs):
        _ = kwargs
        return [{"id": "r1", "text": "alpha"}, {"id": "r2", "text": "beta"}]

    def search_hybrid_with_metrics(self, **kwargs):
        _ = kwargs
        return {
            "rows": [{"id": "r1", "text": "alpha"}, {"id": "r2", "text": "beta"}],
            "duration": {"total_ms": 12.34},
            "profile": {"lanes": ["bm25", "dense", "sparse"]},
            "constraint_hits": 1,
        }

    def api(self, function_name, payload):
        if function_name == "search_bm25":
            return [{"id": "bm25_1", "text": payload.get("query", "")}]
        return {"ok": True}

    def upload_document(
        self,
        *,
        file_path,
        collection_hash_id: str,
        scan_text: bool,
        create_embeddings: bool,
        create_sparse_embeddings: bool,
        await_embedding: bool,
        sparse_embedding_dimensions: int,
    ):
        path_str = str(file_path)
        if path_str.endswith("bad.txt"):
            raise RuntimeError("simulated upload failure")
        self.uploaded.append(path_str)
        return {"hash_id": "doc_demo", "collection_hash_id": collection_hash_id}


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


def test_cli_rag_upload_dir_success(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home4"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    root = tmp_path / "docs"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "b.txt").write_text("b", encoding="utf-8")
    (sub / "note.md").write_text("m", encoding="utf-8")

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

    code = cli.main(
        [
            "rag",
            "upload-dir",
            "--collection-id",
            "col_1",
            "--dir",
            str(root),
            "--pattern",
            "*.txt",
            "--recursive",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"uploaded\": 2" in out
    assert "\"failed\": 0" in out


def test_cli_rag_upload_dir_fail_fast(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home5"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    root = tmp_path / "docs2"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "bad.txt").write_text("x", encoding="utf-8")

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

    code = cli.main(
        [
            "rag",
            "upload-dir",
            "--collection-id",
            "col_2",
            "--dir",
            str(root),
            "--pattern",
            "*.txt",
            "--fail-fast",
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "\"failed\": 1" in out
    assert "simulated upload failure" in out


def test_cli_rag_list_collections(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home6"
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

    code = cli.main(["rag", "list-collections", "--global-collections", "--organization-id", "7"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\"hash_id\": \"col_1\"" in out
    assert "\"organization_id\": 7" in out


def test_cli_rag_list_documents_and_count_chunks(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home7"
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

    code = cli.main(
        [
            "rag",
            "list-documents",
            "--collection-id",
            "col_99",
            "--limit",
            "10",
            "--offset",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"collection_hash_id\": \"col_99\"" in out
    assert "\"count\": 1" in out
    assert "\"limit_used\": 10" in out
    assert "\"offset_used\": 5" in out

    code = cli.main(["rag", "count-chunks", "--collection-ids", "col_99,col_100"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\"chunk_count\": 42" in out
    assert "\"col_99\"" in out
    assert "\"col_100\"" in out


def test_cli_rag_search_with_metrics(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home8"
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

    code = cli.main(
        [
            "rag",
            "search",
            "--collection-id",
            "col_1",
            "--query",
            "main contribution",
            "--with-metrics",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"duration\"" in out
    assert "\"total_ms\": 12.34" in out
    assert "\"constraint_hits\": 1" in out
