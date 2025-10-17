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
