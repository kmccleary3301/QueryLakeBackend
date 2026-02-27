from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, Union

from QueryLake.runtime.retrieval_primitives_legacy import (
    BM25RetrieverParadeDB,
    CrossEncoderReranker,
    DenseRetrieverPGVector,
    FileChunkBM25RetrieverSQL,
    RRFusion,
    SparseRetrieverPGVector,
    WeightedScoreFusion,
)
from QueryLake.typing.config import AuthType
from QueryLake.typing.retrieval_primitives import RetrievalPipelineSpec


RetrieverBuilder = Callable[..., Any]
_RETRIEVER_BUILDERS: Dict[str, RetrieverBuilder] = {}
_DEFAULT_RETRIEVERS_REGISTERED = False


def register_retriever_builder(primitive_id: str, builder: RetrieverBuilder) -> None:
    assert isinstance(primitive_id, str) and len(primitive_id) > 0, "primitive_id must be non-empty"
    _RETRIEVER_BUILDERS[primitive_id] = builder


def list_retriever_builders() -> Dict[str, RetrieverBuilder]:
    _ensure_default_retrievers()
    return dict(_RETRIEVER_BUILDERS)


def _ensure_default_retrievers() -> None:
    global _DEFAULT_RETRIEVERS_REGISTERED
    if _DEFAULT_RETRIEVERS_REGISTERED:
        return
    register_retriever_builder(
        "BM25RetrieverParadeDB",
        lambda **kwargs: BM25RetrieverParadeDB(
            database=kwargs["database"],
            auth=kwargs["auth"],
            search_bm25_fn=kwargs.get("search_bm25_fn"),
        ),
    )
    register_retriever_builder(
        "DenseRetrieverPGVector",
        lambda **kwargs: DenseRetrieverPGVector(
            database=kwargs["database"],
            auth=kwargs["auth"],
            toolchain_function_caller=kwargs["toolchain_function_caller"],
            search_hybrid_fn=kwargs.get("search_hybrid_fn"),
        ),
    )
    register_retriever_builder(
        "SparseRetrieverPGVector",
        lambda **kwargs: SparseRetrieverPGVector(
            database=kwargs["database"],
            auth=kwargs["auth"],
            toolchain_function_caller=kwargs["toolchain_function_caller"],
            search_hybrid_fn=kwargs.get("search_hybrid_fn"),
        ),
    )
    register_retriever_builder(
        "FileChunkBM25RetrieverSQL",
        lambda **kwargs: FileChunkBM25RetrieverSQL(
            database=kwargs["database"],
            auth=kwargs["auth"],
            search_file_chunks_fn=kwargs.get("search_file_chunks_fn"),
        ),
    )
    _DEFAULT_RETRIEVERS_REGISTERED = True


def build_retrievers_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    database,
    auth: AuthType,
    toolchain_function_caller: Optional[Callable[[Any], Union[Callable, Awaitable[Callable]]]] = None,
    search_bm25_fn: Optional[Callable[..., Any]] = None,
    search_hybrid_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
    search_file_chunks_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    _ensure_default_retrievers()
    retrievers: Dict[str, Any] = {}
    for stage in pipeline.stages:
        if not stage.enabled:
            continue
        builder = _RETRIEVER_BUILDERS.get(str(stage.primitive_id))
        if builder is None:
            available = ", ".join(sorted(_RETRIEVER_BUILDERS.keys()))
            raise ValueError(f"Unsupported retriever primitive_id={stage.primitive_id}; available={available}")
        retrievers[stage.stage_id] = builder(
            database=database,
            auth=auth,
            toolchain_function_caller=toolchain_function_caller,
            search_bm25_fn=search_bm25_fn,
            search_hybrid_fn=search_hybrid_fn,
            search_file_chunks_fn=search_file_chunks_fn,
        )
    return retrievers


def select_fusion_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    options: Dict[str, Any],
):
    fusion_primitive = str(
        options.get(
            "fusion_primitive",
            (pipeline.flags or {}).get("fusion_primitive", "WeightedScoreFusion"),
        )
    ).strip().lower()
    if fusion_primitive in {"none", "off", "disabled"}:
        return None
    if fusion_primitive in {"rrfusion", "rrf"}:
        return RRFusion(k=int(options.get("rrf_k", (pipeline.flags or {}).get("rrf_k", 60))))
    return WeightedScoreFusion(
        default_normalization=str(
            options.get(
                "fusion_normalization",
                (pipeline.flags or {}).get("fusion_normalization", "minmax"),
            )
        )
    )


def select_reranker_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    auth: AuthType,
    toolchain_function_caller: Optional[Callable[[Any], Union[Callable, Awaitable[Callable]]]],
    options: Dict[str, Any],
):
    flags = pipeline.flags or {}
    rerank_enabled = bool(options.get("rerank_enabled", flags.get("rerank_enabled", False)))
    if not rerank_enabled:
        return None
    reranker_primitive = str(options.get("reranker_primitive", flags.get("reranker_primitive", "CrossEncoderReranker")))
    if reranker_primitive.strip().lower() in {"none", "off", "disabled"}:
        return None
    if toolchain_function_caller is None:
        raise ValueError("Reranker requires toolchain_function_caller")
    return CrossEncoderReranker(auth=auth, toolchain_function_caller=toolchain_function_caller)
