from __future__ import annotations

import json

import pytest

import querylake_sdk.cli as cli


class _FakeClient:
    last_base_url = None
    last_search_hybrid_kwargs = None
    last_search_hybrid_with_metrics_kwargs = None

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

    def fetch_collection(self, *, collection_hash_id):
        return {"hash_id": str(collection_hash_id), "name": "demo collection"}

    def modify_collection(self, *, collection_hash_id, title=None, description=None):
        return {
            "ok": True,
            "collection_hash_id": str(collection_hash_id),
            "title": title,
            "description": description,
        }

    def count_chunks(self, *, collection_ids=None):
        return {
            "chunk_count": 42,
            "collection_ids": list(collection_ids) if collection_ids is not None else None,
        }

    def search_hybrid(self, **kwargs):
        _FakeClient.last_search_hybrid_kwargs = dict(kwargs)
        return [{"id": "r1", "text": "alpha"}, {"id": "r2", "text": "beta"}]

    def search_hybrid_with_metrics(self, **kwargs):
        _FakeClient.last_search_hybrid_with_metrics_kwargs = dict(kwargs)
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

    def delete_document(self, *, document_hash_id):
        return {"ok": True, "document_hash_id": str(document_hash_id)}

    def get_random_chunks(self, *, limit=5, collection_ids=None):
        out = []
        for idx in range(int(limit)):
            out.append(
                {
                    "id": f"chunk_{idx}",
                    "text": "sample",
                    "collection_ids": list(collection_ids) if collection_ids is not None else None,
                }
            )
        return out

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


def test_cli_rag_upload_dir_dry_run_with_filters(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home5b"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    root = tmp_path / "docs3"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "bad.txt").write_text("x", encoding="utf-8")
    (root / "note.md").write_text("m", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "nested.txt").write_text("n", encoding="utf-8")

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
            "col_3",
            "--dir",
            str(root),
            "--pattern",
            "*",
            "--recursive",
            "--extensions",
            ".txt",
            "--exclude-glob",
            "bad*",
            "--exclude-glob",
            "sub/*",
            "--dry-run",
            "--list-files",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"dry_run\": true" in out.lower()
    assert "\"uploaded\": 0" in out
    assert "\"failed\": 0" in out
    assert "a.txt" in out
    assert "bad.txt" not in out
    assert "nested.txt" not in out
    assert "note.md" not in out


def test_cli_rag_upload_dir_extension_and_exclude_upload(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home5c"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "CONFIG_DIR", home / ".querylake")
    monkeypatch.setattr(cli, "CONFIG_PATH", home / ".querylake" / "sdk_profiles.json")
    monkeypatch.setattr(cli, "QueryLakeClient", _FakeClient)

    root = tmp_path / "docs4"
    root.mkdir(parents=True, exist_ok=True)
    (root / "keep.txt").write_text("k", encoding="utf-8")
    (root / "skip.md").write_text("m", encoding="utf-8")
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
            "col_4",
            "--dir",
            str(root),
            "--pattern",
            "*",
            "--extensions",
            "txt",
            "--exclude-glob",
            "bad*",
            "--list-files",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"requested_files\": 1" in out
    assert "\"uploaded\": 1" in out
    assert "\"failed\": 0" in out
    assert "keep.txt" in out
    assert "bad.txt" not in out
    assert "skip.md" not in out


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


def test_cli_rag_get_and_update_collection(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home8a"
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

    code = cli.main(["rag", "get-collection", "--collection-id", "col_55"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\"hash_id\": \"col_55\"" in out

    code = cli.main(
        [
            "rag",
            "update-collection",
            "--collection-id",
            "col_55",
            "--title",
            "renamed",
            "--description",
            "updated",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"collection_id\": \"col_55\"" in out
    assert "\"title\": \"renamed\"" in out
    assert "\"description\": \"updated\"" in out


def test_cli_rag_update_collection_requires_field(monkeypatch, tmp_path):
    home = tmp_path / "home8b"
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
    with pytest.raises(SystemExit):
        cli.main(["rag", "update-collection", "--collection-id", "col_55"])


def test_cli_rag_random_chunks(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home8c"
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
    code = cli.main(["rag", "random-chunks", "--limit", "3", "--collection-ids", "a,b"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\"count\": 3" in out
    assert "\"chunk_0\"" in out
    assert "\"a\"" in out
    assert "\"b\"" in out


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
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs is not None
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs["limit_bm25"] == 12
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs["limit_similarity"] == 12
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs["limit_sparse"] == 0


def test_cli_rag_search_gate_failure(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home8_gate"
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
            "--min-total-results",
            "3",
        ]
    )
    assert code == 2
    out = capsys.readouterr().out
    assert "\"gate_failed\": true" in out.lower()


def test_cli_rag_search_preset_tri_lane_and_override(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home9"
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
            "multihop query",
            "--preset",
            "tri-lane",
            "--limit-sparse",
            "30",
        ]
    )
    assert code == 0
    _ = capsys.readouterr().out
    assert _FakeClient.last_search_hybrid_kwargs is not None
    assert _FakeClient.last_search_hybrid_kwargs["limit_bm25"] == 12
    assert _FakeClient.last_search_hybrid_kwargs["limit_similarity"] == 12
    assert _FakeClient.last_search_hybrid_kwargs["limit_sparse"] == 30
    assert _FakeClient.last_search_hybrid_kwargs["sparse_weight"] == 0.2


def test_cli_rag_delete_document_requires_confirmation(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home10"
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

    with pytest.raises(SystemExit):
        cli.main(["rag", "delete-document", "--document-id", "doc_42"])

    code = cli.main(["rag", "delete-document", "--document-id", "doc_42", "--yes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\"deleted_document_id\": \"doc_42\"" in out


def test_cli_rag_search_batch(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home11"
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

    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("# comment\nfirst query\n\nsecond query\nthird query\n", encoding="utf-8")
    output_file = tmp_path / "out" / "batch_results.json"

    code = cli.main(
        [
            "rag",
            "search-batch",
            "--collection-id",
            "col_1",
            "--queries-file",
            str(queries_file),
            "--preset",
            "tri-lane",
            "--with-metrics",
            "--max-queries",
            "2",
            "--output-file",
            str(output_file),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "\"query_count\": 2" in out
    assert "\"avg_result_count\": 2.0" in out
    assert "\"avg_duration_ms\": 12.34" in out
    assert "\"first query\"" in out
    assert "\"second query\"" in out
    assert "\"third query\"" not in out
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs is not None
    assert _FakeClient.last_search_hybrid_with_metrics_kwargs["limit_sparse"] == 12
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["query_count"] == 2


def test_cli_rag_search_batch_gate_failure(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home11_gate"
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

    queries_file = tmp_path / "queries_gate.txt"
    queries_file.write_text("first query\nsecond query\n", encoding="utf-8")

    code = cli.main(
        [
            "rag",
            "search-batch",
            "--collection-id",
            "col_1",
            "--queries-file",
            str(queries_file),
            "--min-total-results",
            "3",
        ]
    )
    assert code == 2
    out = capsys.readouterr().out
    assert "\"gate_failed\": true" in out.lower()
    assert "\"queries_below_min_total_results\"" in out
