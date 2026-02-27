from __future__ import annotations

from typing import Any, Dict, Optional

from QueryLake.typing.retrieval_primitives import RetrievalPipelineSpec


def _effective_fusion(
    *,
    options: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    primitive_raw = options.get("fusion_primitive", flags.get("fusion_primitive", "WeightedScoreFusion"))
    primitive = str(primitive_raw or "WeightedScoreFusion").strip()
    primitive_lower = primitive.lower()

    if primitive_lower in {"none", "off", "disabled"}:
        return {
            "enabled": False,
            "primitive": "none",
        }
    if primitive_lower in {"rrf", "rrfusion"}:
        return {
            "enabled": True,
            "primitive": "RRFusion",
            "rrf_k": int(options.get("rrf_k", flags.get("rrf_k", 60))),
        }
    return {
        "enabled": True,
        "primitive": "WeightedScoreFusion",
        "normalization": str(options.get("fusion_normalization", flags.get("fusion_normalization", "minmax"))),
        "weights": options.get("fusion_weights"),
        "score_keys": options.get("fusion_score_keys"),
    }


def _effective_reranker(
    *,
    options: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    rerank_enabled = bool(options.get("rerank_enabled", flags.get("rerank_enabled", False)))
    reranker_primitive = str(options.get("reranker_primitive", flags.get("reranker_primitive", "CrossEncoderReranker")))
    return {
        "enabled": rerank_enabled and reranker_primitive.lower().strip() not in {"none", "off", "disabled"},
        "primitive": reranker_primitive,
        "query_text": options.get("rerank_query_text"),
    }


def build_retrieval_plan_explain(
    *,
    route: str,
    pipeline: RetrievalPipelineSpec,
    options: Optional[Dict[str, Any]] = None,
    pipeline_resolution: Optional[Dict[str, Any]] = None,
    lane_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    opts = dict(options or {})
    flags = dict(pipeline.flags or {})

    return {
        "route": str(route),
        "pipeline": {
            "pipeline_id": pipeline.pipeline_id,
            "pipeline_version": pipeline.version,
            "source": (pipeline_resolution or {}).get("source"),
            "resolution": dict(pipeline_resolution or {}),
            "flags": flags,
            "budgets": dict(pipeline.budgets or {}),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "primitive_id": stage.primitive_id,
                    "enabled": bool(stage.enabled),
                    "config": dict(stage.config or {}),
                }
                for stage in pipeline.stages
            ],
        },
        "effective": {
            "fusion": _effective_fusion(options=opts, flags=flags),
            "reranker": _effective_reranker(options=opts, flags=flags),
            "limit": int(opts.get("limit", 0)),
            "limits": {
                "limit_bm25": opts.get("limit_bm25"),
                "limit_similarity": opts.get("limit_similarity"),
                "limit_sparse": opts.get("limit_sparse"),
            },
            "lane_state": dict(lane_state or {}),
        },
    }

