from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime import retrieval_pipeline_runtime as runtime
from QueryLake.typing.retrieval_primitives import RetrievalPipelineSpec, RetrievalPipelineStage


class _DummyDB:
    pass


def _spec(pipeline_id: str, version: str, *, flags=None) -> RetrievalPipelineSpec:
    return RetrievalPipelineSpec(
        pipeline_id=pipeline_id,
        version=version,
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB")],
        flags=flags or {},
    )


def test_default_pipeline_for_route_has_expected_stage_keys():
    pipeline = runtime.default_pipeline_for_route("search_hybrid")
    assert pipeline is not None
    assert pipeline.pipeline_id == "orchestrated.search_hybrid"
    stage_ids = [stage.stage_id for stage in pipeline.stages]
    assert "bm25" in stage_ids
    assert "dense" in stage_ids
    assert "sparse" in stage_ids
    bm25_stage = next(stage for stage in pipeline.stages if stage.stage_id == "bm25")
    assert bm25_stage.config.get("limit_key") == "limit_bm25"
    assert pipeline.flags.get("fusion_primitive") == "WeightedScoreFusion"
    assert pipeline.flags.get("fusion_normalization") == "minmax"
    assert pipeline.flags.get("rrf_k") == 60


def test_resolve_runtime_pipeline_prefers_override(monkeypatch):
    monkeypatch.setattr(runtime, "resolve_active_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "register_pipeline_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "set_active_pipeline", lambda *args, **kwargs: None)

    def _fetch(_db, *, pipeline_id, version=None):
        if pipeline_id == "override.pipeline" and version == "v9":
            return SimpleNamespace(spec_json=_spec("override.pipeline", "v9").model_dump())
        return None

    monkeypatch.setattr(runtime, "fetch_pipeline_spec", _fetch)

    pipeline, meta = runtime.resolve_runtime_pipeline(
        _DummyDB(),
        route="search_hybrid",
        pipeline_override={"pipeline_id": "override.pipeline", "pipeline_version": "v9"},
        created_by="tester",
    )
    assert pipeline.pipeline_id == "override.pipeline"
    assert pipeline.version == "v9"
    assert meta["source"] == "override"


def test_resolve_runtime_pipeline_uses_binding_when_available(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "resolve_active_pipeline",
        lambda *args, **kwargs: SimpleNamespace(active_pipeline_id="bound.pipeline", active_pipeline_version="v3"),
    )
    monkeypatch.setattr(runtime, "register_pipeline_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "set_active_pipeline", lambda *args, **kwargs: None)

    def _fetch(_db, *, pipeline_id, version=None):
        if pipeline_id == "bound.pipeline" and version == "v3":
            return SimpleNamespace(spec_json=_spec("bound.pipeline", "v3").model_dump())
        return None

    monkeypatch.setattr(runtime, "fetch_pipeline_spec", _fetch)

    pipeline, meta = runtime.resolve_runtime_pipeline(
        _DummyDB(),
        route="search_hybrid",
        created_by="tester",
    )
    assert pipeline.pipeline_id == "bound.pipeline"
    assert pipeline.version == "v3"
    assert meta["source"] == "binding"


def test_resolve_runtime_pipeline_binding_can_carry_tenant_fusion_defaults(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "resolve_active_pipeline",
        lambda *args, **kwargs: SimpleNamespace(active_pipeline_id="bound.pipeline", active_pipeline_version="v3"),
    )
    monkeypatch.setattr(runtime, "register_pipeline_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "set_active_pipeline", lambda *args, **kwargs: None)

    def _fetch(_db, *, pipeline_id, version=None):
        if pipeline_id == "bound.pipeline" and version == "v3":
            return SimpleNamespace(
                spec_json=_spec(
                    "bound.pipeline",
                    "v3",
                    flags={"fusion_primitive": "rrf", "rrf_k": 77, "fusion_normalization": "zscore"},
                ).model_dump()
            )
        return None

    monkeypatch.setattr(runtime, "fetch_pipeline_spec", _fetch)

    pipeline, meta = runtime.resolve_runtime_pipeline(
        _DummyDB(),
        route="search_hybrid",
        tenant_scope="tenant_a",
        created_by="tester",
    )
    assert pipeline.flags.get("fusion_primitive") == "rrf"
    assert pipeline.flags.get("rrf_k") == 77
    assert meta["source"] == "binding"


def test_resolve_runtime_pipeline_falls_back_and_auto_binds(monkeypatch):
    calls = {"register": 0, "bind": 0}
    monkeypatch.setattr(runtime, "resolve_active_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "fetch_pipeline_spec", lambda *args, **kwargs: None)

    def _register(*args, **kwargs):
        calls["register"] += 1
        return None

    def _bind(*args, **kwargs):
        calls["bind"] += 1
        return None

    monkeypatch.setattr(runtime, "register_pipeline_spec", _register)
    monkeypatch.setattr(runtime, "set_active_pipeline", _bind)

    pipeline, meta = runtime.resolve_runtime_pipeline(
        _DummyDB(),
        route="search_bm25.document_chunk",
        created_by="tester",
    )
    assert pipeline.pipeline_id == "orchestrated.search_bm25.document_chunk"
    assert meta["source"] == "default"
    assert calls["register"] == 1
    assert calls["bind"] == 1
