from __future__ import annotations

import json

import httpx
import pytest

from querylake_sdk import QueryLakeAPIError, QueryLakeClient


def _mock_client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.Client(base_url="http://testserver", transport=transport)


def test_api_unwraps_success_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/fetch_all_collections"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["auth"]["oauth2"] == "tok_123"
        return httpx.Response(200, json={"success": True, "result": {"collections": []}})

    client = QueryLakeClient(base_url="http://testserver", oauth2="tok_123")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        result = client.api("fetch_all_collections")
        assert result == {"collections": []}
    finally:
        client.close()


def test_api_raises_on_success_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": False, "error": "bad auth", "trace": "stack..."},
        )

    client = QueryLakeClient(base_url="http://testserver")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        with pytest.raises(QueryLakeAPIError) as exc:
            client.api("search_hybrid", {"query": "hello"})
        assert "bad auth" in str(exc.value)
        assert exc.value.trace == "stack..."
    finally:
        client.close()


def test_upload_document_round_trip(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload_document"
        params = dict(request.url.params)
        assert "parameters" in params
        decoded = json.loads(params["parameters"])
        assert decoded["collection_hash_id"] == "abc123"
        assert decoded["create_embeddings"] is True
        assert decoded["auth"]["oauth2"] == "token"
        body = request.content
        assert b"sample.txt" in body
        return httpx.Response(200, json={"success": True, "result": {"hash_id": "doc1"}})

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        result = client.upload_document(file_path=sample, collection_hash_id="abc123")
        assert result["hash_id"] == "doc1"
    finally:
        client.close()


def test_upload_directory_dry_run_with_filters(tmp_path):
    root = tmp_path / "bulk"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.md").write_text("b", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "c.txt").write_text("c", encoding="utf-8")

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    try:
        payload = client.upload_directory(
            collection_hash_id="abc123",
            directory=root,
            pattern="*",
            recursive=True,
            include_extensions=[".txt"],
            exclude_globs=["sub/*"],
            dry_run=True,
        )
        assert payload["dry_run"] is True
        assert payload["requested_files"] == 1
        assert payload["uploaded"] == 0
        assert payload["failed"] == 0
        assert len(payload["selected_files"]) == 1
        assert payload["selected_files"][0].endswith("a.txt")
    finally:
        client.close()


def test_upload_directory_explicit_file_list_and_errors(tmp_path, monkeypatch):
    root = tmp_path / "bulk2"
    root.mkdir(parents=True, exist_ok=True)
    good = root / "good.txt"
    good.write_text("good", encoding="utf-8")
    bad = root / "bad.txt"
    bad.write_text("bad", encoding="utf-8")

    calls = []

    def _fake_upload_document(**kwargs):
        file_path = str(kwargs["file_path"])
        calls.append(file_path)
        if file_path.endswith("bad.txt"):
            raise RuntimeError("simulated failure")
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _fake_upload_document)
    try:
        payload = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[good, bad],
            fail_fast=False,
        )
        assert payload["selection_mode"] == "explicit-file-list"
        assert payload["requested_files"] == 2
        assert payload["uploaded"] == 1
        assert payload["failed"] == 1
        assert len(payload["errors"]) == 1
        assert "bad.txt" in payload["errors"][0]["file"]
        assert len(calls) == 2
    finally:
        client.close()


def test_search_hybrid_accepts_orchestrated_dict_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/search_hybrid"
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "rows": [{"id": "row1", "text": "hello world"}],
                    "duration": {"total": 0.01},
                },
            },
        )

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        rows = client.search_hybrid(query="hello", collection_ids=["c1"])
        assert len(rows) == 1
        assert rows[0]["id"] == "row1"
        payload = client.search_hybrid_with_metrics(query="hello", collection_ids=["c1"])
        assert isinstance(payload.get("rows"), list)
        assert payload.get("duration", {}).get("total") == 0.01
    finally:
        client.close()


def test_delete_document_uses_hash_id_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/delete_document"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["hash_id"] == "doc_123"
        return httpx.Response(200, json={"success": True, "result": True})

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        result = client.delete_document(document_hash_id="doc_123")
        assert result is True
    finally:
        client.close()


def test_modify_collection_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/modify_document_collection"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["collection_hash_id"] == "col_12"
        assert payload["title"] == "new title"
        assert payload["description"] == "new desc"
        return httpx.Response(200, json={"success": True, "result": True})

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        result = client.modify_collection(
            collection_hash_id="col_12",
            title="new title",
            description="new desc",
        )
        assert result is True
    finally:
        client.close()
