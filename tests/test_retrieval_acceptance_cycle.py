from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "retrieval_acceptance_cycle.py"
    spec = importlib.util.spec_from_file_location("retrieval_acceptance_cycle", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_derive_latency_windows_from_all_runs_prefers_pipeline_id_prefixes():
    mod = _load_module()
    rows = [
        {"run_id": "1", "pipeline_id": "legacy.search_hybrid"},
        {"run_id": "2", "pipeline_id": "orchestrated.search_hybrid"},
        {"run_id": "3", "pipeline_id": "legacy.search_bm25.document_chunk"},
        {"run_id": "4", "pipeline_id": "orchestrated.search_file_chunks"},
        {"run_id": "5", "pipeline_id": "custom.pipeline"},
        {"run_id": "6"},
    ]
    baseline, candidate = mod._derive_latency_windows_from_all_runs(rows)
    assert [row["run_id"] for row in baseline] == ["1", "3"]
    assert [row["run_id"] for row in candidate] == ["2", "4"]


def test_load_rows_supports_list_and_wrapped_rows(tmp_path: Path):
    mod = _load_module()

    list_path = tmp_path / "rows_list.json"
    list_path.write_text(json.dumps([{"run_id": "a"}, "bad", {"run_id": "b"}]), encoding="utf-8")
    wrapped_path = tmp_path / "rows_wrapped.json"
    wrapped_path.write_text(json.dumps({"rows": [{"run_id": "c"}, None]}), encoding="utf-8")
    invalid_path = tmp_path / "rows_invalid.json"
    invalid_path.write_text(json.dumps({"not_rows": []}), encoding="utf-8")

    assert [row["run_id"] for row in mod._load_rows(list_path)] == ["a", "b"]
    assert [row["run_id"] for row in mod._load_rows(wrapped_path)] == ["c"]
    assert mod._load_rows(invalid_path) == []

