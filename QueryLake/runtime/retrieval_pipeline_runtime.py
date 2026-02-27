from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sqlmodel import Session

from QueryLake.runtime.retrieval_pipeline_registry import (
    fetch_pipeline_spec,
    register_pipeline_spec,
)
from QueryLake.runtime.retrieval_rollout import resolve_active_pipeline, set_active_pipeline
from QueryLake.typing.retrieval_primitives import (
    RetrievalPipelineSpec,
    RetrievalPipelineStage,
)


def default_pipeline_for_route(route: str) -> Optional[RetrievalPipelineSpec]:
    route = str(route or "").strip()
    if route == "search_hybrid":
        return RetrievalPipelineSpec(
            pipeline_id="orchestrated.search_hybrid",
            version="v1",
            stages=[
                RetrievalPipelineStage(
                    stage_id="bm25",
                    primitive_id="BM25RetrieverParadeDB",
                    config={
                        "table": "document_chunk",
                        "sort_by": "score",
                        "sort_dir": "DESC",
                        "group_chunks": False,
                        "limit_key": "limit_bm25",
                        "offset_key": "offset_bm25",
                        "bm25_query_text_key": "bm25_query_text",
                    },
                ),
                RetrievalPipelineStage(
                    stage_id="dense",
                    primitive_id="DenseRetrieverPGVector",
                    config={
                        "group_chunks": False,
                        "limit_key": "limit_similarity",
                        "dense_query_text_key": "dense_query_text",
                    },
                ),
                RetrievalPipelineStage(
                    stage_id="sparse",
                    primitive_id="SparseRetrieverPGVector",
                    config={
                        "group_chunks": False,
                        "limit_key": "limit_sparse",
                        "sparse_query_text_key": "sparse_query_text",
                        "sparse_query_value_key": "sparse_query_value",
                        "sparse_embedding_function": "embedding_sparse",
                        "sparse_dimensions": 1024,
                    },
                ),
            ],
            flags={
                "fusion_primitive": "WeightedScoreFusion",
                "fusion_normalization": "minmax",
                "rrf_k": 60,
            },
        )
    if route == "search_bm25.document_chunk":
        return RetrievalPipelineSpec(
            pipeline_id="orchestrated.search_bm25.document_chunk",
            version="v1",
            stages=[
                RetrievalPipelineStage(
                    stage_id="bm25",
                    primitive_id="BM25RetrieverParadeDB",
                    config={
                        "table": "document_chunk",
                        "group_chunks": False,
                        "limit_key": "limit",
                        "offset_key": "offset",
                        "bm25_query_text_key": "bm25_query_text",
                    },
                )
            ],
        )
    if route == "search_file_chunks":
        return RetrievalPipelineSpec(
            pipeline_id="orchestrated.search_file_chunks",
            version="v1",
            stages=[
                RetrievalPipelineStage(
                    stage_id="file_bm25",
                    primitive_id="FileChunkBM25RetrieverSQL",
                    config={
                        "limit_key": "limit",
                        "offset_key": "offset",
                        "bm25_query_text_key": "bm25_query_text",
                        "sort_by": "score",
                        "sort_dir": "DESC",
                    },
                )
            ],
        )
    return None


def _validate_spec(spec_json: Dict[str, Any]) -> RetrievalPipelineSpec:
    return RetrievalPipelineSpec.model_validate(spec_json)


def _fetch_validated(
    database: Session,
    *,
    pipeline_id: str,
    pipeline_version: str,
) -> Optional[RetrievalPipelineSpec]:
    row = fetch_pipeline_spec(database, pipeline_id=pipeline_id, version=pipeline_version)
    if row is None or not isinstance(row.spec_json, dict):
        return None
    return _validate_spec(row.spec_json)


def resolve_runtime_pipeline(
    database: Session,
    *,
    route: str,
    tenant_scope: Optional[str] = None,
    created_by: Optional[str] = None,
    pipeline_override: Optional[Dict[str, str]] = None,
    fallback_pipeline: Optional[RetrievalPipelineSpec] = None,
    auto_bind_default: bool = True,
) -> Tuple[RetrievalPipelineSpec, Dict[str, Any]]:
    route = str(route or "").strip()
    metadata: Dict[str, Any] = {
        "source": "unknown",
        "route": route,
        "tenant_scope": tenant_scope,
        "notes": [],
    }

    override_id = None
    override_version = None
    if isinstance(pipeline_override, dict):
        override_id = pipeline_override.get("pipeline_id")
        override_version = pipeline_override.get("pipeline_version")
    if isinstance(override_id, str) and isinstance(override_version, str) and override_id and override_version:
        try:
            resolved = _fetch_validated(
                database,
                pipeline_id=override_id,
                pipeline_version=override_version,
            )
            if resolved is not None:
                metadata["source"] = "override"
                metadata["pipeline_id"] = resolved.pipeline_id
                metadata["pipeline_version"] = resolved.version
                return resolved, metadata
            metadata["notes"].append("override_pipeline_not_found")
        except Exception:
            metadata["notes"].append("override_lookup_failed")

    binding = None
    try:
        binding = resolve_active_pipeline(database, route=route, tenant_scope=tenant_scope)
    except Exception:
        metadata["notes"].append("binding_lookup_failed")

    if binding is not None:
        try:
            resolved = _fetch_validated(
                database,
                pipeline_id=binding.active_pipeline_id,
                pipeline_version=binding.active_pipeline_version,
            )
            if resolved is not None:
                metadata["source"] = "binding"
                metadata["pipeline_id"] = resolved.pipeline_id
                metadata["pipeline_version"] = resolved.version
                return resolved, metadata
            metadata["notes"].append("binding_target_pipeline_missing_or_invalid")
        except Exception:
            metadata["notes"].append("binding_target_lookup_failed")

    fallback = fallback_pipeline or default_pipeline_for_route(route)
    if fallback is None:
        raise ValueError(f"No fallback retrieval pipeline configured for route={route}")

    try:
        register_pipeline_spec(
            database,
            pipeline_id=fallback.pipeline_id,
            version=fallback.version,
            spec_json=fallback.model_dump(),
            created_by=created_by,
            md={"route": route, "auto_registered": True},
        )
    except Exception:
        metadata["notes"].append("fallback_registration_failed")

    if auto_bind_default:
        try:
            set_active_pipeline(
                database,
                route=route,
                tenant_scope=tenant_scope,
                pipeline_id=fallback.pipeline_id,
                pipeline_version=fallback.version,
                updated_by=created_by,
                reason="auto-init default retrieval pipeline",
            )
        except Exception:
            metadata["notes"].append("fallback_binding_failed")

    metadata["source"] = "default"
    metadata["pipeline_id"] = fallback.pipeline_id
    metadata["pipeline_version"] = fallback.version
    return fallback, metadata
