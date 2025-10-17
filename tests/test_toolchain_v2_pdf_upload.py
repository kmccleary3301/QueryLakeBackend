from pathlib import Path
import sys
import json
import asyncio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.typing.toolchains import ToolChainV2
from QueryLake.runtime.session import ToolchainSessionV2


def test_v2_pdf_upload_toolchain_flow():
    # Load the JSON toolchain definition
    path = ROOT / "toolchains" / "demo_pdf_upload_v2.json"
    data = json.loads(path.read_text())
    tool = ToolChainV2.model_validate(data)

    # Prepare a dummy umbrella and capture calls
    calls = {}

    async def files_process_version(database=None, auth=None, umbrella=None, file_id: str = None, version_id: str = None):
        calls["args"] = {"file_id": file_id, "version_id": version_id}
        return {"job_id": "jb_test", "status": "COMPLETED"}

    class DummyUmbrella:
        def api_function_getter(self, name: str):
            assert name == "files_process_version"
            return files_process_version

    sess = ToolchainSessionV2(
        session_id="sess_pdf",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": DummyUmbrella()},
        emit_event=lambda kind, payload, meta: None,
        job_registry=None,
    )

    # Simulate client: call attach_pdf with a file meta after uploading via /files
    file_meta = {"file_id": "fl_123", "version_id": "fv_001", "logical_name": "sample.pdf", "bytes_cas": "abcd"}

    async def run():
        await sess.process_event("attach_pdf", {"file_meta": file_meta}, actor="tester")

    asyncio.run(run())

    # Check the file has been stored in session.files
    assert "uploads" in sess.files
    assert isinstance(sess.files["uploads"], list)
    assert sess.files["uploads"][0]["file_id"] == "fl_123"

    # Check API node was invoked and state updated
    assert calls["args"] == {"file_id": "fl_123", "version_id": "fv_001"}
    assert sess.state["last_job"]["job_id"] == "jb_test"
    assert sess.state["last_job"]["status"] == "COMPLETED"

