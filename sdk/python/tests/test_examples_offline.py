from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_bulk_ingest_example_offline_demo(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "a.md").write_text("# Demo\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "examples" / "sdk" / "rag_bulk_ingest_and_search.py"
    sdk_src = repo_root / "sdk" / "python" / "src"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(sdk_src)
        if not existing_pythonpath
        else f"{sdk_src}:{existing_pythonpath}"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--offline-demo",
            "--dir",
            str(docs_dir),
            "--pattern",
            "*.md",
            "--query",
            "demo query",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["collection_id"] == "offline_demo_collection"
    assert payload["upload"]["uploaded"] == 1
    assert payload["_meta"]["offline_demo"] is True
    assert len(payload["results"]) >= 1
