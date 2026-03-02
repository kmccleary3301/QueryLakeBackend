from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from querylake_sdk import (
    HybridSearchOptions,
    QueryLakeAPIError,
    QueryLakeClient,
    UploadDirectoryOptions,
)


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


def test_upload_document_injects_idempotency_key(tmp_path):
    sample = tmp_path / "sample2.txt"
    sample.write_text("hello world", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload_document"
        params = dict(request.url.params)
        decoded = json.loads(params["parameters"])
        metadata = decoded.get("document_metadata")
        assert isinstance(metadata, dict)
        assert metadata.get("existing") == "value"
        ingest_meta = metadata.get("_querylake_ingest")
        assert isinstance(ingest_meta, dict)
        assert ingest_meta.get("idempotency_key") == "idem_123"
        return httpx.Response(200, json={"success": True, "result": {"hash_id": "doc2"}})

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    client._client.close()
    client._client = _mock_client(handler)
    try:
        result = client.upload_document(
            file_path=sample,
            collection_hash_id="abc123",
            document_metadata={"existing": "value"},
            idempotency_key="idem_123",
        )
        assert result["hash_id"] == "doc2"
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


def test_upload_directory_dedupe_content_hash_run_local(tmp_path, monkeypatch):
    root = tmp_path / "bulk_hash"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "a.txt"
    one.write_text("same", encoding="utf-8")
    two = root / "b.txt"
    two.write_text("same", encoding="utf-8")
    three = root / "c.txt"
    three.write_text("different", encoding="utf-8")

    calls = []

    def _fake_upload_document(**kwargs):
        calls.append(str(kwargs["file_path"]))
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _fake_upload_document)
    try:
        payload = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[one, two, three],
            dedupe_by_content_hash=True,
            dedupe_scope="run-local",
        )
        assert payload["uploaded"] == 2
        assert payload["failed"] == 0
        assert payload["dedupe_by_content_hash"] is True
        assert payload["dedupe_scope"] == "run-local"
        assert payload["dedupe_skipped"] == 1
        assert len(payload["dedupe_skipped_files"]) == 1
        assert payload["dedupe_skipped_files"][0]["reason"] == "run-local-duplicate"
        assert calls == [str(one), str(three)]
    finally:
        client.close()


def test_upload_directory_dedupe_checkpoint_resume(tmp_path, monkeypatch):
    root = tmp_path / "bulk_hash_checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("same", encoding="utf-8")
    one_hash = hashlib.sha256(one.read_bytes()).hexdigest()
    checkpoint_file = tmp_path / "checkpoint_hash.json"
    checkpoint_file.write_text(
        json.dumps(
            {
                "version": 1,
                "selection_sha256": "mismatch-ok",
                "uploaded_files": [],
                "uploaded_content_hashes": [one_hash],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def _fake_upload_document(**kwargs):
        calls.append(str(kwargs["file_path"]))
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _fake_upload_document)
    try:
        payload = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[one],
            checkpoint_file=checkpoint_file,
            resume=True,
            strict_checkpoint_match=False,
            dedupe_by_content_hash=True,
            dedupe_scope="checkpoint-resume",
        )
        assert payload["dedupe_skipped"] == 1
        assert payload["uploaded"] == 0
        assert payload["failed"] == 0
        assert payload["status"] == "already_complete"
        assert payload["dedupe_skipped_files"][0]["reason"] == "checkpoint-resume-duplicate"
        assert calls == []
    finally:
        client.close()


def test_upload_directory_content_hash_idempotency_strategy(tmp_path, monkeypatch):
    root = tmp_path / "bulk_idem"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("one", encoding="utf-8")
    one_hash = hashlib.sha256(one.read_bytes()).hexdigest()

    keys = []

    def _fake_upload_document(**kwargs):
        keys.append(kwargs.get("idempotency_key"))
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _fake_upload_document)
    try:
        payload = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[one],
            idempotency_strategy="content-hash",
            idempotency_prefix="test-prefix",
        )
        assert payload["uploaded"] == 1
        assert keys == [f"test-prefix:abc123:{one_hash}"]
    finally:
        client.close()


def test_upload_directory_checkpoint_resume(tmp_path, monkeypatch):
    root = tmp_path / "bulk3"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("one", encoding="utf-8")
    two = root / "two.txt"
    two.write_text("two", encoding="utf-8")
    checkpoint_file = tmp_path / "checkpoint.json"

    first_pass_calls = []

    def _first_pass_upload(**kwargs):
        file_path = str(kwargs["file_path"])
        first_pass_calls.append(file_path)
        if file_path.endswith("two.txt"):
            raise RuntimeError("transient failure")
        return {"hash_id": "doc_ok"}

    second_pass_calls = []

    def _second_pass_upload(**kwargs):
        file_path = str(kwargs["file_path"])
        second_pass_calls.append(file_path)
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _first_pass_upload)
    try:
        first = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[one, two],
            fail_fast=True,
            checkpoint_file=checkpoint_file,
            checkpoint_save_every=1,
        )
        assert first["uploaded"] == 1
        assert first["failed"] == 1
        assert checkpoint_file.exists()

        monkeypatch.setattr(client, "upload_document", _second_pass_upload)
        second = client.upload_directory(
            collection_hash_id="abc123",
            file_paths=[one, two],
            checkpoint_file=checkpoint_file,
            resume=True,
        )
        assert second["resumed_from_checkpoint"] is True
        assert second["skipped_already_uploaded"] == 1
        assert second["uploaded"] == 1
        assert second["failed"] == 0
        assert first_pass_calls == [str(one), str(two)]
        assert second_pass_calls == [str(two)]
    finally:
        client.close()


def test_upload_directory_rejects_invalid_dedupe_or_idempotency(tmp_path):
    root = tmp_path / "bulk5"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("one", encoding="utf-8")

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    try:
        with pytest.raises(ValueError, match="Unsupported dedupe_scope"):
            client.upload_directory(
                collection_hash_id="abc123",
                file_paths=[one],
                dedupe_by_content_hash=True,
                dedupe_scope="bad-scope",
            )
        with pytest.raises(ValueError, match="Unsupported idempotency_strategy"):
            client.upload_directory(
                collection_hash_id="abc123",
                file_paths=[one],
                idempotency_strategy="bad-strategy",
            )
    finally:
        client.close()


def test_upload_directory_options_validation():
    with pytest.raises(ValueError, match="checkpoint_save_every"):
        UploadDirectoryOptions(file_paths=["/tmp/a.txt"], checkpoint_save_every=0)
    with pytest.raises(ValueError, match="dedupe_scope"):
        UploadDirectoryOptions(file_paths=["/tmp/a.txt"], dedupe_scope="invalid")
    with pytest.raises(ValueError, match="idempotency_strategy"):
        UploadDirectoryOptions(file_paths=["/tmp/a.txt"], idempotency_strategy="invalid")
    with pytest.raises(ValueError, match="resume=True requires checkpoint_file"):
        UploadDirectoryOptions(file_paths=["/tmp/a.txt"], resume=True)


def test_upload_directory_with_options(tmp_path, monkeypatch):
    root = tmp_path / "bulk_options"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("one", encoding="utf-8")

    calls = []

    def _fake_upload_document(**kwargs):
        calls.append(str(kwargs["file_path"]))
        return {"hash_id": "doc_ok"}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", _fake_upload_document)
    try:
        options = UploadDirectoryOptions(
            file_paths=[one],
            dedupe_by_content_hash=True,
            idempotency_strategy="content-hash",
            idempotency_prefix="typed",
        )
        payload = client.upload_directory_with_options(
            collection_hash_id="abc123",
            options=options,
        )
        assert payload["uploaded"] == 1
        assert payload["failed"] == 0
        assert payload["dedupe_by_content_hash"] is True
        assert calls == [str(one)]
    finally:
        client.close()


def test_upload_directory_checkpoint_hash_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "bulk4"
    root.mkdir(parents=True, exist_ok=True)
    one = root / "one.txt"
    one.write_text("one", encoding="utf-8")
    checkpoint_file = tmp_path / "checkpoint_mismatch.json"
    checkpoint_file.write_text(
        json.dumps(
            {
                "version": 1,
                "selection_sha256": "mismatch",
                "uploaded_files": [],
            }
        ),
        encoding="utf-8",
    )

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "upload_document", lambda **kwargs: {"hash_id": "doc_ok"})
    try:
        with pytest.raises(ValueError, match="selection hash mismatch"):
            client.upload_directory(
                collection_hash_id="abc123",
                file_paths=[one],
                checkpoint_file=checkpoint_file,
                resume=True,
            )
    finally:
        client.close()


def test_search_hybrid_with_option_models(monkeypatch):
    captured: list[dict] = []

    def _fake_api(function_name, payload, **kwargs):
        assert function_name == "search_hybrid"
        captured.append(dict(payload))
        return {"rows": [{"id": "row1", "text": "hello"}], "duration": {"total_ms": 1.0}}

    client = QueryLakeClient(base_url="http://testserver", oauth2="token")
    monkeypatch.setattr(client, "api", _fake_api)
    try:
        options = HybridSearchOptions(
            limit_bm25=5,
            limit_similarity=7,
            limit_sparse=9,
            bm25_weight=0.3,
            similarity_weight=0.4,
            sparse_weight=0.3,
        )
        rows = client.search_hybrid_with_options(
            query="hello",
            collection_ids=["c1"],
            options=options,
        )
        metrics = client.search_hybrid_with_metrics_options(
            query="hello",
            collection_ids=["c1"],
            options=options,
        )
        assert len(rows) == 1
        assert metrics["duration"]["total_ms"] == 1.0
        assert captured[0]["limit_bm25"] == 5
        assert captured[0]["limit_similarity"] == 7
        assert captured[0]["limit_sparse"] == 9
        assert captured[0]["bm25_weight"] == 0.3
        assert captured[0]["similarity_weight"] == 0.4
        assert captured[0]["sparse_weight"] == 0.3
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
