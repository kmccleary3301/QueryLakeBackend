from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime import retrieval_primitive_factory as factory
from QueryLake.runtime.retrieval_primitives_legacy import RRFusion, WeightedScoreFusion
from QueryLake.typing.retrieval_primitives import RetrievalPipelineSpec, RetrievalPipelineStage


def test_default_retriever_builders_registered():
    builders = factory.list_retriever_builders()
    assert "BM25RetrieverParadeDB" in builders
    assert "DenseRetrieverPGVector" in builders
    assert "SparseRetrieverPGVector" in builders
    assert "FileChunkBM25RetrieverSQL" in builders


def test_custom_retriever_builder_registration(monkeypatch):
    calls = {"count": 0}

    def _build_custom(**kwargs):
        calls["count"] += 1
        return {"custom": True}

    factory.register_retriever_builder("CustomRetriever", _build_custom)
    pipeline = RetrievalPipelineSpec(
        pipeline_id="custom.pipeline",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="custom", primitive_id="CustomRetriever")],
    )

    retrievers = factory.build_retrievers_for_pipeline(
        pipeline=pipeline,
        database=object(),
        auth={"username": "tester", "password_prehash": "x"},
        toolchain_function_caller=None,
        search_bm25_fn=lambda **kwargs: [],
        search_hybrid_fn=None,
        search_file_chunks_fn=None,
    )

    assert "custom" in retrievers
    assert retrievers["custom"] == {"custom": True}
    assert calls["count"] == 1


def test_select_fusion_for_pipeline_respects_rrf_override():
    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_hybrid",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB")],
        flags={"fusion_primitive": "WeightedScoreFusion", "fusion_normalization": "minmax", "rrf_k": 60},
    )
    fusion = factory.select_fusion_for_pipeline(
        pipeline=pipeline,
        options={"fusion_primitive": "rrf", "rrf_k": 42},
    )
    assert isinstance(fusion, RRFusion)
    assert fusion.k == 42


def test_select_fusion_for_pipeline_uses_pipeline_defaults_when_no_override():
    pipeline = RetrievalPipelineSpec(
        pipeline_id="tenant.bound",
        version="v2",
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB")],
        flags={"fusion_primitive": "WeightedScoreFusion", "fusion_normalization": "zscore"},
    )
    fusion = factory.select_fusion_for_pipeline(
        pipeline=pipeline,
        options={},
    )
    assert isinstance(fusion, WeightedScoreFusion)
    assert fusion.default_normalization == "zscore"
