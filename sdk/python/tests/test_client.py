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
