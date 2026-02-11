from pathlib import Path
import os
import shutil
import asyncio

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.files.object_store import LocalCASObjectStore
from QueryLake.files.service import FilesRuntimeService
from QueryLake.runtime.sse import SessionStreamHub


class _DummyDB:
    def add(self, _row):
        return None

    def commit(self):
        return None


class _DummyUmbrella:
    def __init__(self, handles=None):
        self.chandra_handles = handles or {}


def test_local_cas_store_roundtrip(tmp_path: Path):
    base = tmp_path / "cas"
    store = LocalCASObjectStore(base_dir=str(base))
    data = b"hello world"
    cas = store.put_bytes(data)
    assert store.exists(cas)
    out = store.get_bytes(cas)
    assert out == data


def test_files_fingerprint_stability():
    fp1 = FilesRuntimeService.compute_fingerprint("deadbeef")
    fp2 = FilesRuntimeService.compute_fingerprint("deadbeef")
    assert fp1 == fp2
    fp3 = FilesRuntimeService.compute_fingerprint("deadbeee")
    assert fp1 != fp3


def test_sse_hub_publish_and_backlog_replay():
    async def run():
        hub = SessionStreamHub()
        session_id = "file_x"
        sub = await hub.subscribe(session_id)
        # Push a couple of events
        await sub.push({"event": "FILE_UPLOADED", "data": "{}", "id": 1})
        await sub.push({"event": "CHUNKED", "data": "{}", "id": 2})

        # Consume
        items = []
        async def consume(n=2):
            async for item in sub.stream():
                items.append(item)
                if len(items) >= n:
                    break

        await consume(2)
        assert [i["event"] for i in items] == ["FILE_UPLOADED", "CHUNKED"]

        await hub.unsubscribe(session_id, sub)

    asyncio.run(run())


def test_render_cache_key_is_stable_and_page_sensitive():
    key_1 = FilesRuntimeService._compute_render_cache_key("abc123", dpi=144, page_num=1)
    key_2 = FilesRuntimeService._compute_render_cache_key("abc123", dpi=144, page_num=1)
    key_3 = FilesRuntimeService._compute_render_cache_key("abc123", dpi=144, page_num=2)
    assert key_1 == key_2
    assert key_1 != key_3


def test_chandra_handle_selection_prefers_configured_id(monkeypatch):
    handles = {
        "chandra-v1": "HANDLE_ID",
        "Chandra Friendly Name": "HANDLE_ALIAS",
    }
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella(handles))

    monkeypatch.setenv("QUERYLAKE_DEFAULT_CHANDRA_ID", "chandra-v1")
    picked = asyncio.run(service._select_chandra_handle())
    assert picked == "HANDLE_ID"


def test_chandra_handle_selection_falls_back_to_id_like_key(monkeypatch):
    handles = {
        "chandra-v2": "HANDLE_ID",
        "Chandra Friendly Name": "HANDLE_ALIAS",
    }
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella(handles))

    monkeypatch.delenv("QUERYLAKE_DEFAULT_CHANDRA_ID", raising=False)
    picked = asyncio.run(service._select_chandra_handle())
    assert picked == "HANDLE_ID"


def test_render_cache_eviction_order(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_CHANDRA_RENDER_CACHE_MAX_ENTRIES", "2")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    async def run():
        await service._render_cache_set("k1", "v1")
        await service._render_cache_set("k2", "v2")
        await service._render_cache_set("k3", "v3")
        first = await service._render_cache_get("k1")
        second = await service._render_cache_get("k2")
        third = await service._render_cache_get("k3")
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first is None
    assert second == "v2"
    assert third == "v3"


def test_pdf_text_layer_auto_mode_selects_digital_like_pages(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "auto")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE", "50")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_COVERAGE", "0.6")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    pages = [
        "A" * 120,
        "B" * 90,
        "C" * 80,
        "D" * 70,
        "",
    ]
    decision = service._evaluate_pdf_text_layer_candidate(pages)
    assert decision["selected"] is True
    assert decision["reason"] == "auto_threshold_met"
    assert decision["pages"] == 5


def test_pdf_text_layer_auto_mode_rejects_sparse_text(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "auto")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE", "80")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_COVERAGE", "0.8")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    pages = [
        "A" * 100,
        "B" * 40,
        "",
        "",
        "C" * 30,
    ]
    decision = service._evaluate_pdf_text_layer_candidate(pages)
    assert decision["selected"] is False
    assert decision["reason"] == "auto_threshold_miss"


def test_pdf_text_layer_prefer_mode_selects_when_any_text_exists(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "prefer")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    decision = service._evaluate_pdf_text_layer_candidate(["", "hello", ""])
    assert decision["selected"] is True
    assert decision["reason"] == "prefer_nonempty"


def test_pdf_text_layer_off_mode_never_selects(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "off")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    decision = service._evaluate_pdf_text_layer_candidate(["x" * 100, "y" * 100])
    assert decision["selected"] is False
    assert decision["reason"] == "mode_off"


def test_pdf_text_layer_mixed_mode_selects_page_overrides(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "mixed")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE", "50")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    decision = service._evaluate_pdf_text_layer_page_overrides(
        ["A" * 120, "B" * 40, "", "C" * 75]
    )
    assert decision["selected"] is True
    assert decision["mode"] == "mixed"
    assert decision["selected_pages"] == 2
    assert decision["selected_page_indices"] == [0, 3]


def test_pdf_text_layer_mixed_mode_handles_zero_selected(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "mixed")
    monkeypatch.setenv("QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE", "200")
    service = FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(), umbrella=_DummyUmbrella())

    decision = service._evaluate_pdf_text_layer_page_overrides(
        ["A" * 120, "B" * 40, "", "C" * 75]
    )
    assert decision["selected"] is False
    assert decision["selected_pages"] == 0
    assert decision["selected_page_indices"] == []
