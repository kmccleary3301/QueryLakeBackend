import asyncio
import concurrent.futures
import json
import os
import uuid
import time
import random
import math
from typing import Optional, Dict
from sqlmodel import Field, Session, select, func
from sqlalchemy import Column, DDL, event, text, Index, JSON, MetaData, Table
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from typing import List, Tuple, Union, Literal, Callable, Awaitable, Any
from pgvector.sqlalchemy import Vector
import pgvector
from pydantic import BaseModel
import random
import time
import re
from ..misc_functions.paradedb_query_parser import parse_search
from ..database.sql_db_tables import (
    DocumentChunk,
    document_raw,
    document_segment as SegmentTable,
    retrieval_run as RetrievalRunTable,
    CHUNK_INDEXED_COLUMNS,
    DOCUMENT_INDEXED_COLUMNS,
    SEGMENT_INDEXED_COLUMNS,
)
from ..database.create_db_session import configured_sparse_index_dimensions
from ..database.sql_db_tables import file_chunk as FileChunkTable, file_version as FileVersionTable, file as FileTable, FILE_CHUNK_INDEXED_COLUMNS as FILE_FIELDS
from ..typing.config import AuthType
from .single_user_auth import get_user
from .collections import assert_collections_priviledge
from ..misc_functions.toolchain_state_management import safe_serialize
from ..observability import metrics
from ..runtime.retrieval_cache import (
    build_embedding_cache_key,
    build_rerank_cache_key,
    get_embedding,
    get_rerank_score,
    set_embedding,
    set_rerank_score,
)
from ..runtime.retrieval_runs import log_retrieval_run
from ..runtime.rate_limiter import acquire_concurrency, release_concurrency
from ..runtime.request_context import get_request_id
from ..runtime.retrieval_orchestrator import PipelineOrchestrator
from ..runtime.retrieval_pipeline_runtime import (
    default_pipeline_for_route,
    resolve_runtime_pipeline,
)
from ..runtime.retrieval_explain import build_retrieval_plan_explain
from ..runtime.retrieval_primitive_factory import (
    build_retrievers_for_pipeline as factory_build_retrievers_for_pipeline,
    select_fusion_for_pipeline as factory_select_fusion_for_pipeline,
    select_reranker_for_pipeline as factory_select_reranker_for_pipeline,
)
from ..runtime.retrieval_primitives_legacy import (
    AdjacentChunkPacker,
    CitationAwarePacker,
    DiversityAwarePacker,
    TokenBudgetPacker,
)
from ..typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalPipelineStage,
    RetrievalPipelineSpec,
    RetrievalRequest,
)

ADAPTIVE_PROFILE_DEFAULT = "natural_language"
ADAPTIVE_LANE_POLICY_DEFAULTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "identifier_exact": {
        "weights": {"bm25": 0.72, "dense": 0.18, "sparse": 0.10},
        "limits": {"bm25": 24, "dense": 8, "sparse": 8},
    },
    "constraint_lexical": {
        "weights": {"bm25": 0.62, "dense": 0.23, "sparse": 0.15},
        "limits": {"bm25": 20, "dense": 10, "sparse": 10},
    },
    "natural_language": {
        "weights": {"bm25": 0.35, "dense": 0.50, "sparse": 0.15},
        "limits": {"bm25": 10, "dense": 16, "sparse": 10},
    },
}

class DocumentChunkDictionary(BaseModel):
    id: Union[str, int, List[str], List[int]]
    creation_timestamp: float
    collection_type: Optional[Union[str, None]]
    document_id: Optional[Union[str, None]]
    document_chunk_number: Optional[Union[int, Tuple[int, int], None]]
    # document_integrity: Optional[Union[str, None]]
    collection_id: Optional[Union[str, None]]
    document_name: str
    # website_url : Optional[Union[str, None]]
    # private: bool
    md: dict
    document_md: dict
    text: str
    embedding: Optional[List[float]] = None
    
    hybrid_score: Optional[float] = None
    bm25_score: Optional[float] = None
    similarity_score: Optional[float] = None
    sparse_score: Optional[float] = None
    
class DocumentChunkDictionaryReranked(DocumentChunkDictionary):
    rerank_score: float
    
class DocumentRawDictionary(BaseModel):
    id: Optional[str]
    file_name: str
    creation_timestamp: float
    integrity_sha256: str
    size_bytes: int
    encryption_key_secure: Optional[str]
    document_collection_id: Optional[str] = None
    website_url: Optional[str]
    blob_id: Optional[str]
    blob_dir: Optional[str]
    finished_processing: int
    md: dict
    bm25_score: Optional[float] = None

chunk_dict_arguments = [
    "id", 
    "creation_timestamp", 
    "collection_type", 
    "document_id", 
    "document_chunk_number", 
    "document_integrity", 
    "collection_id", 
    "document_name", 
    "website_url", 
    "private", 
    "md", 
    "document_md", 
    "text", 
    "rerank_score"
]

segment_dict_arguments = [
    "id",
    "creation_timestamp",
    "collection_type",
    "document_id",
    "document_chunk_number",
    "document_integrity",
    "collection_id",
    "document_name",
    "website_url",
    "private",
    "md",
    "document_md",
    "text",
]

document_dict_arguments = [
    "id",
    "file_name",
    "creation_timestamp",
    "integrity_sha256",
    "size_bytes",
    "encryption_key_secure",
    "document_collection_id",
    "website_url",
    "blob_id",
    "blob_dir",
    "finished_processing",
    "md"
]

document_collection_attrs = [
    "document_collection_id"
]
    

field_strings_no_rerank = [e for e in chunk_dict_arguments if e not in ["rerank_score"]]
column_attributes = [getattr(DocumentChunk, e) for e in field_strings_no_rerank]
retrieved_fields_string = ", ".join([f"{DocumentChunk.__tablename__}."+e for e in field_strings_no_rerank])
retrieved_fields_string_bm25 = ", ".join(field_strings_no_rerank)
retrieved_segment_fields_string = (
    f"{SegmentTable.__tablename__}.id, "
    f"{SegmentTable.__tablename__}.created_at AS creation_timestamp, "
    f"NULL::text AS collection_type, "
    f"NULL::text AS document_id, "
    f"{SegmentTable.__tablename__}.segment_index AS document_chunk_number, "
    f"NULL::text AS document_integrity, "
    f"COALESCE({SegmentTable.__tablename__}.md->>'collection_id', '') AS collection_id, "
    f"COALESCE({SegmentTable.__tablename__}.md->>'document_name', '') AS document_name, "
    f"NULL::text AS website_url, "
    f"FALSE AS private, "
    f"{SegmentTable.__tablename__}.md AS md, "
    f"'{{}}'::jsonb AS document_md, "
    f"{SegmentTable.__tablename__}.text AS text"
)

document_field_strings = [e for e in document_dict_arguments]
retrieved_document_fields_string = ", ".join([f"{document_raw.__tablename__}."+e for e in document_field_strings])



def convert_chunk_query_result(query_results: tuple, rerank: bool = False, return_wrapped : bool = False):
    wrapped_args =  {chunk_dict_arguments[i]: query_results[i] for i in range(min(len(query_results), len(chunk_dict_arguments)))}
    if return_wrapped:
        return wrapped_args
    try:
        return DocumentChunkDictionary(**wrapped_args) if not rerank else DocumentChunkDictionaryReranked(**wrapped_args)
    except Exception as e:
        print("Error with result tuple:", query_results)
        print("Error with wrapped args:", wrapped_args)
        raise e
    
def convert_doc_query_result(query_results: tuple, return_wrapped : bool = False):
    wrapped_args =  {document_dict_arguments[i]: query_results[i] for i in range(min(len(query_results), len(document_dict_arguments)))}
    if return_wrapped:
        return wrapped_args
    try:
        return DocumentRawDictionary(**wrapped_args)
    except Exception as e:
        print("Error with result tuple:", query_results)
        print("Error with wrapped args:", wrapped_args)
        raise e


def convert_segment_query_result(query_results: tuple, return_wrapped: bool = False):
    wrapped_args = {
        segment_dict_arguments[i]: query_results[i]
        for i in range(min(len(query_results), len(segment_dict_arguments)))
    }
    if return_wrapped:
        return wrapped_args
    return DocumentChunkDictionary(**wrapped_args)


def _extract_dense_embedding_payload(payload: Any) -> Optional[List[float]]:
    value = payload
    if isinstance(value, dict):
        for key in ["embedding", "dense", "dense_embedding", "dense_vec", "dense_vecs"]:
            if key in value:
                value = value[key]
                break
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except Exception:
            return None
    return None


def _normalize_sparse_query_value(
    value: Any,
    dimensions: int = 1024,
    *,
    strict_dimensions: bool = False,
    source_label: str = "sparse_query",
):
    current = value
    if isinstance(current, dict):
        sparse_key_found = False
        for key in [
            "sparse",
            "sparse_embedding",
            "sparse_vec",
            "sparse_vecs",
            "lexical_weights",
            "weights",
            "sparse_weights",
        ]:
            if key in current:
                current = current[key]
                sparse_key_found = True
                break
        if not sparse_key_found:
            dense_fallback = _extract_dense_embedding_payload(current)
            if dense_fallback is not None:
                current = dense_fallback

    if current is None:
        return None

    try:
        if isinstance(current, pgvector.SparseVector):
            if current.dimensions() == dimensions:
                return current
            if strict_dimensions:
                raise ValueError(
                    f"Sparse dimension mismatch for {source_label}: expected {dimensions}, observed {current.dimensions()}"
                )
            mapping = {int(idx): float(weight) for idx, weight in zip(current.indices(), current.values())}
            return pgvector.SparseVector(mapping, dimensions)

        if isinstance(current, str):
            parsed = pgvector.SparseVector.from_text(current)
            if parsed.dimensions() != dimensions:
                if strict_dimensions:
                    raise ValueError(
                        f"Sparse dimension mismatch for {source_label}: expected {dimensions}, observed {parsed.dimensions()}"
                    )
                mapping = {int(idx): float(weight) for idx, weight in zip(parsed.indices(), parsed.values())}
                return pgvector.SparseVector(mapping, dimensions)
            return parsed

        if isinstance(current, dict) and "indices" in current and "values" in current:
            indices = current.get("indices") or []
            values = current.get("values") or []
            dim = int(current.get("dimensions", dimensions))
            if strict_dimensions and dim != dimensions:
                raise ValueError(
                    f"Sparse dimension mismatch for {source_label}: expected {dimensions}, observed {dim}"
                )
            mapping = {}
            for idx, weight in zip(indices, values):
                if weight is None:
                    continue
                mapping[int(idx)] = float(weight)
            return pgvector.SparseVector(mapping, dim)

        if isinstance(current, dict):
            mapping = {}
            for key, weight in current.items():
                if weight is None:
                    continue
                try:
                    idx = int(key)
                    val = float(weight)
                except Exception:
                    continue
                if val != 0.0:
                    mapping[idx] = val
            return pgvector.SparseVector(mapping, dimensions)

        if isinstance(current, tuple):
            current = list(current)

        if isinstance(current, list):
            if strict_dimensions and len(current) > 0 and len(current) != dimensions:
                raise ValueError(
                    f"Sparse dimension mismatch for {source_label}: expected {dimensions}, observed {len(current)}"
                )
            mapping = {}
            for idx, weight in enumerate(current):
                try:
                    val = float(weight)
                except Exception:
                    continue
                if val != 0.0:
                    mapping[idx] = val
            return pgvector.SparseVector(mapping, dimensions)
    except ValueError as exc:
        if "Sparse dimension mismatch" in str(exc):
            raise
        return None
    except Exception:
        return None

    return None


def _resolve_adaptive_lane_policy(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    policy: Dict[str, Dict[str, Dict[str, float]]] = {
        profile: {
            "weights": dict(cfg.get("weights", {})),
            "limits": dict(cfg.get("limits", {})),
        }
        for profile, cfg in ADAPTIVE_LANE_POLICY_DEFAULTS.items()
    }
    if not isinstance(overrides, dict):
        return policy

    for profile_name, profile_cfg in overrides.items():
        if not isinstance(profile_name, str) or not isinstance(profile_cfg, dict):
            continue
        target_profile = policy.setdefault(
            profile_name,
            {
                "weights": dict(policy[ADAPTIVE_PROFILE_DEFAULT]["weights"]),
                "limits": dict(policy[ADAPTIVE_PROFILE_DEFAULT]["limits"]),
            },
        )
        weights = profile_cfg.get("weights", {})
        limits = profile_cfg.get("limits", {})
        if isinstance(weights, dict):
            for key in ("bm25", "dense", "sparse"):
                if key in weights:
                    try:
                        target_profile["weights"][key] = max(0.0, float(weights[key]))
                    except Exception:
                        continue
        if isinstance(limits, dict):
            for key in ("bm25", "dense", "sparse"):
                if key in limits:
                    try:
                        target_profile["limits"][key] = max(0, min(200, int(limits[key])))
                    except Exception:
                        continue
    return policy


def _infer_adaptive_query_profile(query_text: str) -> str:
    text_value = str(query_text or "")
    trimmed = text_value.strip()
    if len(trimmed) == 0:
        return ADAPTIVE_PROFILE_DEFAULT
    lowered = trimmed.lower()
    token_count = len(re.findall(r"\w+", trimmed))
    has_quotes = '"' in trimmed
    has_operators = bool(re.search(r'(^|\s)([+\-])\w+|(\w+:\w+)|\b(and|or|not)\b', lowered))
    has_identifier_like = bool(
        re.search(r"\b[A-Za-z]+[-_]\d{2,}\b|\b[A-Z]{2,}[-_][A-Z0-9]{2,}\b|\b\d{4,}\b", trimmed)
    )

    if has_identifier_like and token_count <= 10:
        return "identifier_exact"
    if has_quotes or has_operators:
        return "constraint_lexical"
    return ADAPTIVE_PROFILE_DEFAULT


def _sparse_vector_to_entries(vector_value: pgvector.SparseVector) -> List[Tuple[int, float]]:
    coo = vector_value.to_coo()
    cols = getattr(coo, "col", [])
    vals = getattr(coo, "data", [])
    entries: List[Tuple[int, float]] = []
    for idx, val in zip(cols, vals):
        try:
            float_val = float(val)
        except Exception:
            continue
        if float_val != 0.0:
            entries.append((int(idx), float_val))
    return entries


def _apply_sparse_pruning_and_calibration(
    sparse_value: pgvector.SparseVector,
    *,
    max_terms: Optional[int] = None,
    min_abs_weight: float = 0.0,
    calibration: str = "none",
) -> Tuple[pgvector.SparseVector, int, int]:
    entries = _sparse_vector_to_entries(sparse_value)
    terms_before = len(entries)

    if min_abs_weight > 0:
        entries = [(idx, weight) for idx, weight in entries if abs(weight) >= min_abs_weight]

    if isinstance(max_terms, int) and max_terms > 0 and len(entries) > max_terms:
        entries = sorted(entries, key=lambda item: abs(item[1]), reverse=True)[:max_terms]

    calibration_mode = str(calibration or "none").strip().lower()
    if calibration_mode in {"l1", "l2", "maxabs"} and len(entries) > 0:
        values = [weight for _, weight in entries]
        denom = 1.0
        if calibration_mode == "l1":
            denom = sum(abs(v) for v in values)
        elif calibration_mode == "l2":
            denom = math.sqrt(sum((v * v) for v in values))
        else:
            denom = max(abs(v) for v in values)
        if denom > 0:
            entries = [(idx, (weight / denom)) for idx, weight in entries]

    mapping = {idx: float(weight) for idx, weight in entries if float(weight) != 0.0}
    terms_after = len(mapping)
    return pgvector.SparseVector(mapping, sparse_value.dimensions()), terms_before, terms_after


def _default_dynamic_limit_cap(profile_name: str) -> int:
    profile = str(profile_name or ADAPTIVE_PROFILE_DEFAULT).strip().lower()
    if profile == "identifier_exact":
        return 40
    if profile == "constraint_lexical":
        return 36
    return 30


def _apply_dynamic_lane_limit_schedule(
    *,
    limit_bm25: int,
    limit_similarity: int,
    limit_sparse: int,
    bm25_weight: float,
    similarity_weight: float,
    sparse_weight: float,
    use_bm25: bool,
    use_similarity: bool,
    use_sparse: bool,
    total_cap: int,
    min_per_enabled: int = 4,
) -> Tuple[int, int, int, bool]:
    lanes: Dict[str, Dict[str, float]] = {}
    if use_bm25:
        lanes["bm25"] = {"limit": int(max(1, limit_bm25)), "weight": float(max(0.0, bm25_weight))}
    if use_similarity:
        lanes["dense"] = {"limit": int(max(1, limit_similarity)), "weight": float(max(0.0, similarity_weight))}
    if use_sparse:
        lanes["sparse"] = {"limit": int(max(1, limit_sparse)), "weight": float(max(0.0, sparse_weight))}
    if len(lanes) == 0:
        return limit_bm25, limit_similarity, limit_sparse, False

    current_total = sum(int(info["limit"]) for info in lanes.values())
    cap = int(max(len(lanes), total_cap))
    cap = min(cap, current_total)
    if current_total <= cap:
        return limit_bm25, limit_similarity, limit_sparse, False

    min_lane = max(1, int(min_per_enabled))
    weights = {name: max(0.0, float(info["weight"])) for name, info in lanes.items()}
    weight_sum = sum(weights.values())
    if weight_sum <= 1e-12:
        weights = {name: 1.0 for name in lanes.keys()}
        weight_sum = float(len(weights))

    alloc: Dict[str, int] = {}
    for name, info in lanes.items():
        desired = int(round(cap * (weights[name] / weight_sum)))
        desired = max(min_lane, desired)
        alloc[name] = min(int(info["limit"]), desired)

    while sum(alloc.values()) > cap:
        reducible = [name for name in lanes.keys() if alloc[name] > 1]
        if len(reducible) == 0:
            break
        reducible.sort(key=lambda name: (alloc[name], weights[name]), reverse=True)
        alloc[reducible[0]] -= 1

    while sum(alloc.values()) < cap:
        expandable = [name for name, info in lanes.items() if alloc[name] < int(info["limit"])]
        if len(expandable) == 0:
            break
        expandable.sort(
            key=lambda name: ((int(lanes[name]["limit"]) - alloc[name]) * max(weights[name], 1e-6), weights[name]),
            reverse=True,
        )
        alloc[expandable[0]] += 1

    return (
        alloc.get("bm25", 0 if not use_bm25 else max(1, limit_bm25)),
        alloc.get("dense", 0 if not use_similarity else max(1, limit_similarity)),
        alloc.get("sparse", 0 if not use_sparse else max(1, limit_sparse)),
        True,
    )


def _apply_queue_pressure_schedule(
    *,
    queue_utilization: Optional[float],
    limit_bm25: int,
    limit_similarity: int,
    limit_sparse: int,
    bm25_weight: float,
    similarity_weight: float,
    sparse_weight: float,
    use_bm25: bool,
    use_similarity: bool,
    use_sparse: bool,
    min_per_enabled: int,
    soft_utilization: float,
    hard_utilization: float,
    soft_scale: float,
    hard_scale: float,
    disable_sparse_at_hard: bool,
) -> Tuple[int, int, int, bool, str]:
    if queue_utilization is None:
        return limit_bm25, limit_similarity, limit_sparse, False, "none"
    util = float(max(0.0, min(2.0, queue_utilization)))
    soft = float(max(0.1, min(1.0, soft_utilization)))
    hard = float(max(soft, min(1.25, hard_utilization)))
    if util < soft:
        return limit_bm25, limit_similarity, limit_sparse, False, "none"

    base_total = int(limit_bm25 + limit_similarity + limit_sparse)
    if base_total <= 0:
        return limit_bm25, limit_similarity, limit_sparse, False, "none"

    regime = "hard" if util >= hard else "soft"
    scale = float(hard_scale if regime == "hard" else soft_scale)
    scale = max(0.15, min(1.0, scale))
    target_cap = max(1, int(round(float(base_total) * scale)))

    lane_use_sparse = bool(use_sparse)
    lane_limit_sparse = int(limit_sparse)
    lane_sparse_weight = float(sparse_weight)
    if regime == "hard" and disable_sparse_at_hard and lane_use_sparse:
        lane_use_sparse = False
        lane_limit_sparse = 0
        lane_sparse_weight = 0.0
        target_cap = min(target_cap, max(1, int(limit_bm25 + limit_similarity)))

    bm25_new, dense_new, sparse_new, applied = _apply_dynamic_lane_limit_schedule(
        limit_bm25=int(limit_bm25),
        limit_similarity=int(limit_similarity),
        limit_sparse=int(lane_limit_sparse),
        bm25_weight=float(bm25_weight),
        similarity_weight=float(similarity_weight),
        sparse_weight=float(lane_sparse_weight),
        use_bm25=bool(use_bm25),
        use_similarity=bool(use_similarity),
        use_sparse=bool(lane_use_sparse),
        total_cap=int(target_cap),
        min_per_enabled=int(min_per_enabled),
    )
    return int(bm25_new), int(dense_new), int(sparse_new), bool(applied), regime


def find_overlap(string_a: str, string_b: str) -> int:
    max_overlap = min(len(string_a), len(string_b))
    for i in range(max_overlap, 0, -1):
        if string_a[-i:] == string_b[:i]:
            return i
    return 0

def group_adjacent_chunks(chunks: List[DocumentChunkDictionary]) -> List[DocumentChunkDictionary]:
    document_bin : Dict[str, List[DocumentChunkDictionary]] = {}
    for chunk in chunks:
        if chunk.document_id in document_bin:
            document_bin[chunk.document_id].append(chunk)
        else:
            document_bin[chunk.document_id] = [chunk]
    new_results = []
    
    for bin in document_bin:
        if len(document_bin[bin]) == 0:
            continue
        document_bin[bin] = sorted(document_bin[bin], key=lambda x: x.document_chunk_number[0] if isinstance(x.document_chunk_number, tuple) else x.document_chunk_number)
        current_chunk = document_bin[bin][0]
        most_recent_chunk_added = False
        
        for chunk in document_bin[bin][1:]:
            current_chunk_bottom_index = current_chunk.document_chunk_number[0] if isinstance(current_chunk.document_chunk_number, tuple) else current_chunk.document_chunk_number
            current_chunk_top_index = current_chunk.document_chunk_number[1] if isinstance(current_chunk.document_chunk_number, tuple) else current_chunk.document_chunk_number
            chunk_bottom_index = chunk.document_chunk_number[0] if isinstance(chunk.document_chunk_number, tuple) else chunk.document_chunk_number
            chunk_top_index = chunk.document_chunk_number[1] if isinstance(chunk.document_chunk_number, tuple) else chunk.document_chunk_number
            
            
            
            most_recent_chunk_added = False
            if chunk_bottom_index == current_chunk_top_index + 1:
                
                overlap = find_overlap(current_chunk.text, chunk.text)
                
                if overlap > 100:
                    current_chunk.text += chunk.text[overlap:]
                else:
                    current_chunk.text += "\n\n" + chunk.text
                
                current_chunk.document_chunk_number = (current_chunk_bottom_index, chunk_top_index)
                if isinstance(current_chunk.id, (int, str)):
                    current_chunk.id = [current_chunk.id]
                current_chunk.id.append(chunk.id)
                keys_set = ["bm25_score", "similarity_score", "hybrid_score"] + (["rerank_score"] \
                        if isinstance(current_chunk, DocumentChunkDictionaryReranked) and \
                            isinstance(chunk, DocumentChunkDictionaryReranked)
                        else []
                )
                for key in keys_set:
                    if not any([getattr(chunk, key) is None, getattr(current_chunk, key) is None]):
                        max_value = max([0 if e is None else e for e in [getattr(chunk, key), getattr(current_chunk, key)]])
                        setattr(current_chunk, key, max_value)
            else:
                most_recent_chunk_added = True
                new_results.append(current_chunk)
                current_chunk = chunk
        
        if not most_recent_chunk_added:
            new_results.append(current_chunk)
        # if not most_recent_chunk_added:
    return new_results


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return int(value)


def _env_float(name: str, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return float(value)


def _candidate_to_document_chunk(candidate: RetrievalCandidate) -> DocumentChunkDictionary:
    md = candidate.metadata or {}
    row_id: Union[str, List[str]] = candidate.content_id
    merged_ids = md.get("merged_content_ids")
    if isinstance(merged_ids, list) and len(merged_ids) > 0:
        row_id = [str(v) for v in merged_ids]
    creation_timestamp = md.get("creation_timestamp")
    if creation_timestamp is None:
        creation_timestamp = 0.0
    return DocumentChunkDictionary(
        id=row_id,
        creation_timestamp=float(creation_timestamp),
        collection_type=md.get("collection_type"),
        document_id=md.get("document_id"),
        document_chunk_number=md.get("document_chunk_number"),
        collection_id=md.get("collection_id"),
        document_name=md.get("document_name") or "",
        md=md.get("md", {}),
        document_md=md.get("document_md", {}),
        text=candidate.text or "",
        bm25_score=candidate.stage_scores.get("bm25_score"),
        similarity_score=candidate.stage_scores.get("similarity_score"),
        hybrid_score=candidate.stage_scores.get("weighted_fused")
        or candidate.stage_scores.get("rrf_fused")
        or candidate.stage_scores.get("hybrid_score"),
    )


def _candidate_to_file_chunk_result(candidate: RetrievalCandidate) -> Dict[str, Any]:
    md = candidate.metadata or {}
    row: Dict[str, Any] = {
        "id": candidate.content_id,
        "text": candidate.text or "",
        "md": md.get("md", {}),
        "created_at": float(md.get("created_at", 0.0) or 0.0),
        "file_version_id": md.get("file_version_id"),
    }
    score = candidate.stage_scores.get("bm25_score")
    if isinstance(score, (int, float)):
        row["bm25_score"] = float(score)
    return row


def _run_async_sync(coro: Awaitable[Any]) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def _select_packer(*, group_chunks: bool, options: Optional[Dict[str, Any]] = None):
    options = options or {}
    variant = str(
        options.get(
            "packing_mode",
            os.getenv(
                "QUERYLAKE_RETRIEVAL_PACKER_VARIANT",
                "adjacent" if group_chunks else "none",
            ),
        )
    ).strip().lower()

    if variant in {"", "none", "off"} and not group_chunks:
        return None
    if variant in {"", "adjacent"}:
        return AdjacentChunkPacker() if group_chunks or variant == "adjacent" else None
    if variant in {"diversity", "diversity_aware"}:
        return DiversityAwarePacker()
    if variant in {"citation", "citation_aware"}:
        return CitationAwarePacker()
    if variant in {"token", "token_budget"}:
        return TokenBudgetPacker()
    # Fail-safe fallback keeps behavior deterministic and backward-compatible.
    return AdjacentChunkPacker() if group_chunks else None


def _decode_query_payload(query_text: str) -> Union[str, dict]:
    if not isinstance(query_text, str):
        return query_text
    trimmed = query_text.strip()
    if trimmed.startswith("{") or trimmed.startswith("["):
        try:
            decoded = json.loads(trimmed)
            if isinstance(decoded, (dict, list, str)):
                return decoded
        except Exception:
            return query_text
    return query_text


def _extract_quoted_phrases(query_text: str, max_phrases: int = 4) -> List[str]:
    if not isinstance(query_text, str) or len(query_text.strip()) == 0:
        return []
    phrases: List[str] = []
    for match in re.finditer(r'"([^"]+)"', query_text):
        value = (match.group(1) or "").strip()
        if len(value) < 2:
            continue
        value = value[:160]
        if value not in phrases:
            phrases.append(value)
        if len(phrases) >= max(1, int(max_phrases)):
            break
    return phrases


def _infer_tenant_scope(auth: AuthType) -> Optional[str]:
    if isinstance(auth, dict):
        for key in ("tenant_scope", "organization_id", "org_id", "workspace_id"):
            value = auth.get(key)
            if isinstance(value, str) and len(value.strip()) > 0:
                return value.strip()
    return None


def _resolve_route_pipeline(
    database: Session,
    *,
    route: str,
    user_name: str,
    auth: AuthType,
    pipeline_override: Optional[Dict[str, str]],
    fallback_pipeline: Optional[RetrievalPipelineSpec],
):
    return resolve_runtime_pipeline(
        database,
        route=route,
        tenant_scope=_infer_tenant_scope(auth),
        created_by=user_name,
        pipeline_override=pipeline_override,
        fallback_pipeline=fallback_pipeline,
        auto_bind_default=True,
    )


def _build_retrievers_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    database: Session,
    auth: AuthType,
    toolchain_function_caller: Optional[Callable[[Any], Union[Callable, Awaitable[Callable]]]],
    search_bm25_fn: Callable[..., Any],
    search_hybrid_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]],
    search_file_chunks_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    return factory_build_retrievers_for_pipeline(
        pipeline=pipeline,
        database=database,
        auth=auth,
        toolchain_function_caller=toolchain_function_caller,
        search_bm25_fn=search_bm25_fn,
        search_hybrid_fn=search_hybrid_fn,
        search_file_chunks_fn=search_file_chunks_fn,
    )


def _select_fusion_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    options: Dict[str, Any],
):
    return factory_select_fusion_for_pipeline(
        pipeline=pipeline,
        options=options,
    )


def _select_reranker_for_pipeline(
    *,
    pipeline: RetrievalPipelineSpec,
    auth: AuthType,
    toolchain_function_caller: Optional[Callable[[Any], Union[Callable, Awaitable[Callable]]]],
    options: Dict[str, Any],
):
    return factory_select_reranker_for_pipeline(
        pipeline=pipeline,
        auth=auth,
        toolchain_function_caller=toolchain_function_caller,
        options=options,
    )
          

async def search_hybrid(
    database: Session,
    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
    auth : AuthType,
    query: Union[str, dict[str, Any]],
    embedding: List[float] = None,
    embedding_sparse: Any = None,
    collection_ids: List[str] = [],
    limit_bm25: int = 10,
    limit_similarity: int = 10,
    limit_sparse: int = 0,
    similarity_weight: float = 0.1,
    bm25_weight: float = 0.9,
    sparse_weight: float = 0.0,
    use_bm25: Optional[bool] = None,
    use_similarity: Optional[bool] = None,
    use_sparse: Optional[bool] = None,
    sparse_embedding_function: str = "embedding_sparse",
    sparse_dimensions: int = 1024,
    strict_constraint_prefilter: bool = True,
    constraint_query_text: Optional[str] = None,
    fusion_primitive: Optional[str] = None,
    fusion_normalization: Optional[str] = None,
    rrf_k: int = 60,
    explain_plan: bool = False,
    return_statement : bool = False,
    web_search : bool = False,
    rerank : bool = False,
    group_chunks : bool = True,
    _direct_stage_call: bool = False,
    _skip_observability: bool = False,
    _pipeline_override: Optional[Dict[str, str]] = None,
    adaptive_lane_routing: bool = False,
    adaptive_query_profile: Optional[str] = None,
    adaptive_lane_policy: Optional[Dict[str, Any]] = None,
    dynamic_lane_budgeting: bool = False,
    dynamic_lane_total_limit_cap: Optional[int] = None,
    dynamic_lane_min_per_enabled: int = 4,
    queue_aware_admission: bool = True,
    queue_admission_concurrency_limit: Optional[int] = None,
    queue_admission_ttl_seconds: Optional[int] = None,
    queue_throttle_enabled: bool = True,
    queue_throttle_soft_utilization: float = 0.75,
    queue_throttle_hard_utilization: float = 0.90,
    queue_throttle_soft_scale: float = 0.75,
    queue_throttle_hard_scale: float = 0.50,
    queue_throttle_disable_sparse_at_hard: bool = True,
    sparse_prune_max_terms: Optional[int] = None,
    sparse_prune_min_abs_weight: float = 0.0,
    sparse_calibration: str = "none",
    bm25_catch_all_fields: Optional[List[str]] = None,
) -> List[DocumentChunkDictionary]:
    # TODO: Check permissions on specified collections.
    t_1 = time.time()
    
    (_, user_auth) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert isinstance(similarity_weight, (float, int)) and isinstance(bm25_weight, (float, int)) and isinstance(sparse_weight, (float, int)), \
        "`similarity_weight`, `bm25_weight`, and `sparse_weight` must be floats"
    assert similarity_weight >= 0 and bm25_weight >= 0 and sparse_weight >= 0, \
        "`similarity_weight`, `bm25_weight`, and `sparse_weight` must be non-negative"

    assert isinstance(limit_bm25, int) and 0 <= limit_bm25 <= 200, \
        "`limit_bm25` must be an int between 0 and 200"
    assert isinstance(limit_similarity, int) and 0 <= limit_similarity <= 200, \
        "`limit_similarity` must be an int between 0 and 200"
    assert isinstance(limit_sparse, int) and 0 <= limit_sparse <= 200, \
        "`limit_sparse` must be an int between 0 and 200"
    assert isinstance(sparse_dimensions, int) and sparse_dimensions > 0, \
        "`sparse_dimensions` must be a positive integer"
    assert isinstance(strict_constraint_prefilter, bool), "`strict_constraint_prefilter` must be boolean"
    assert isinstance(rrf_k, int) and rrf_k > 0, "`rrf_k` must be a positive integer"
    assert isinstance(explain_plan, bool), "`explain_plan` must be boolean"
    assert isinstance(dynamic_lane_budgeting, bool), "`dynamic_lane_budgeting` must be boolean"
    assert isinstance(queue_aware_admission, bool), "`queue_aware_admission` must be boolean"
    assert isinstance(queue_throttle_enabled, bool), "`queue_throttle_enabled` must be boolean"
    assert isinstance(queue_throttle_disable_sparse_at_hard, bool), "`queue_throttle_disable_sparse_at_hard` must be boolean"
    if dynamic_lane_total_limit_cap is not None:
        assert isinstance(dynamic_lane_total_limit_cap, int) and dynamic_lane_total_limit_cap > 0, \
            "`dynamic_lane_total_limit_cap` must be a positive integer or null"
    if queue_admission_concurrency_limit is not None:
        assert isinstance(queue_admission_concurrency_limit, int) and queue_admission_concurrency_limit >= 0, \
            "`queue_admission_concurrency_limit` must be a non-negative integer or null"
    if queue_admission_ttl_seconds is not None:
        assert isinstance(queue_admission_ttl_seconds, int) and queue_admission_ttl_seconds > 0, \
            "`queue_admission_ttl_seconds` must be a positive integer or null"
    assert isinstance(queue_throttle_soft_utilization, (float, int)) and float(queue_throttle_soft_utilization) > 0, \
        "`queue_throttle_soft_utilization` must be > 0"
    assert isinstance(queue_throttle_hard_utilization, (float, int)) and float(queue_throttle_hard_utilization) > 0, \
        "`queue_throttle_hard_utilization` must be > 0"
    assert isinstance(queue_throttle_soft_scale, (float, int)) and 0 < float(queue_throttle_soft_scale) <= 1.0, \
        "`queue_throttle_soft_scale` must be in (0, 1]"
    assert isinstance(queue_throttle_hard_scale, (float, int)) and 0 < float(queue_throttle_hard_scale) <= 1.0, \
        "`queue_throttle_hard_scale` must be in (0, 1]"
    assert isinstance(dynamic_lane_min_per_enabled, int) and dynamic_lane_min_per_enabled > 0, \
        "`dynamic_lane_min_per_enabled` must be a positive integer"
    if bm25_catch_all_fields is not None:
        assert isinstance(bm25_catch_all_fields, list), "`bm25_catch_all_fields` must be a list of field names or null"

    explicit_use_bm25 = use_bm25
    explicit_use_similarity = use_similarity
    explicit_use_sparse = use_sparse

    if sparse_prune_max_terms is not None:
        assert isinstance(sparse_prune_max_terms, int) and sparse_prune_max_terms > 0, \
            "`sparse_prune_max_terms` must be a positive integer or null"
    assert isinstance(sparse_prune_min_abs_weight, (float, int)) and float(sparse_prune_min_abs_weight) >= 0.0, \
        "`sparse_prune_min_abs_weight` must be a non-negative float"
    sparse_calibration = str(sparse_calibration or "none").strip().lower()
    assert sparse_calibration in {"none", "l1", "l2", "maxabs"}, \
        "`sparse_calibration` must be one of: none, l1, l2, maxabs"

    if isinstance(query, str):
        query = {"bm25": query, "embedding": query, "sparse": query}
        if rerank:
            query["rerank"] = query["bm25"]
    else:
        query = dict(query)
        if query.get("bm25") is None:
            fallback_bm25 = query.get("embedding")
            if not isinstance(fallback_bm25, str):
                fallback_bm25 = query.get("sparse") if isinstance(query.get("sparse"), str) else ""
            query["bm25"] = str(fallback_bm25)
        if query.get("embedding") is None:
            query["embedding"] = str(query.get("bm25", ""))
        if query.get("sparse") is None:
            query["sparse"] = query.get("bm25", "")
        if rerank and query.get("rerank") is None:
            query["rerank"] = str(query.get("bm25", ""))

    if isinstance(query, dict):
        if query.get("adaptive_lane_routing") is not None:
            adaptive_lane_routing = bool(query.get("adaptive_lane_routing"))
        if adaptive_query_profile is None and isinstance(query.get("adaptive_query_profile"), str):
            adaptive_query_profile = str(query.get("adaptive_query_profile"))
        if adaptive_lane_policy is None and isinstance(query.get("adaptive_lane_policy"), dict):
            adaptive_lane_policy = query.get("adaptive_lane_policy")
        if sparse_prune_max_terms is None and query.get("sparse_prune_max_terms") is not None:
            try:
                sparse_prune_max_terms = int(query.get("sparse_prune_max_terms"))
            except Exception:
                sparse_prune_max_terms = None
        if query.get("sparse_prune_min_abs_weight") is not None:
            try:
                sparse_prune_min_abs_weight = float(query.get("sparse_prune_min_abs_weight"))
            except Exception:
                pass
        if query.get("sparse_calibration") is not None:
            sparse_calibration = str(query.get("sparse_calibration") or sparse_calibration).strip().lower()
        if query.get("dynamic_lane_budgeting") is not None:
            dynamic_lane_budgeting = bool(query.get("dynamic_lane_budgeting"))
        if dynamic_lane_total_limit_cap is None and query.get("dynamic_lane_total_limit_cap") is not None:
            try:
                dynamic_lane_total_limit_cap = int(query.get("dynamic_lane_total_limit_cap"))
            except Exception:
                dynamic_lane_total_limit_cap = None
        if query.get("dynamic_lane_min_per_enabled") is not None:
            try:
                dynamic_lane_min_per_enabled = int(query.get("dynamic_lane_min_per_enabled"))
            except Exception:
                pass
        if query.get("queue_aware_admission") is not None:
            queue_aware_admission = bool(query.get("queue_aware_admission"))
        if query.get("queue_admission_concurrency_limit") is not None:
            try:
                queue_admission_concurrency_limit = int(query.get("queue_admission_concurrency_limit"))
            except Exception:
                queue_admission_concurrency_limit = None
        if query.get("queue_admission_ttl_seconds") is not None:
            try:
                queue_admission_ttl_seconds = int(query.get("queue_admission_ttl_seconds"))
            except Exception:
                queue_admission_ttl_seconds = None
        if query.get("queue_throttle_enabled") is not None:
            queue_throttle_enabled = bool(query.get("queue_throttle_enabled"))
        if query.get("queue_throttle_soft_utilization") is not None:
            try:
                queue_throttle_soft_utilization = float(query.get("queue_throttle_soft_utilization"))
            except Exception:
                pass
        if query.get("queue_throttle_hard_utilization") is not None:
            try:
                queue_throttle_hard_utilization = float(query.get("queue_throttle_hard_utilization"))
            except Exception:
                pass
        if query.get("queue_throttle_soft_scale") is not None:
            try:
                queue_throttle_soft_scale = float(query.get("queue_throttle_soft_scale"))
            except Exception:
                pass
        if query.get("queue_throttle_hard_scale") is not None:
            try:
                queue_throttle_hard_scale = float(query.get("queue_throttle_hard_scale"))
            except Exception:
                pass
        if query.get("queue_throttle_disable_sparse_at_hard") is not None:
            queue_throttle_disable_sparse_at_hard = bool(query.get("queue_throttle_disable_sparse_at_hard"))
        if query.get("strict_constraint_prefilter") is not None:
            strict_constraint_prefilter = bool(query.get("strict_constraint_prefilter"))
        if constraint_query_text is None and isinstance(query.get("constraints"), str):
            constraint_query_text = str(query.get("constraints"))
        if fusion_primitive is None and isinstance(query.get("fusion_primitive"), str):
            fusion_primitive = str(query.get("fusion_primitive"))
        if fusion_normalization is None and isinstance(query.get("fusion_normalization"), str):
            fusion_normalization = str(query.get("fusion_normalization"))
        if query.get("rrf_k") is not None:
            try:
                rrf_k = int(query.get("rrf_k"))
            except Exception:
                pass
        if query.get("explain_plan") is not None:
            explain_plan = bool(query.get("explain_plan"))
        if bm25_catch_all_fields is None and isinstance(query.get("bm25_catch_all_fields"), list):
            bm25_catch_all_fields = list(query.get("bm25_catch_all_fields"))

    resolved_catch_all_fields = ["text"]
    if isinstance(bm25_catch_all_fields, list):
        candidate_fields = []
        for field_name in bm25_catch_all_fields:
            if not isinstance(field_name, str):
                continue
            field_clean = field_name.strip()
            if field_clean in CHUNK_INDEXED_COLUMNS:
                candidate_fields.append(field_clean)
        if len(candidate_fields) > 0:
            resolved_catch_all_fields = list(dict.fromkeys(candidate_fields))
    assert sparse_calibration in {"none", "l1", "l2", "maxabs"}, \
        "`sparse_calibration` must be one of: none, l1, l2, maxabs"

    adaptive_lane_applied = False
    adaptive_policy_source = "none"
    adaptive_profile_selected = str(adaptive_query_profile).strip() if isinstance(adaptive_query_profile, str) and len(str(adaptive_query_profile).strip()) > 0 else ""
    dynamic_budget_applied = False
    dynamic_budget_cap = 0
    queue_admission_applied = False
    queue_admission_limit_resolved = int(
        queue_admission_concurrency_limit
        if isinstance(queue_admission_concurrency_limit, int)
        else _env_int("QUERYLAKE_SEARCH_CONCURRENCY_LIMIT", 0, minimum=0, maximum=10000)
    )
    queue_admission_ttl_resolved = int(
        queue_admission_ttl_seconds
        if isinstance(queue_admission_ttl_seconds, int)
        else _env_int("QUERYLAKE_SEARCH_CONCURRENCY_TTL_SECONDS", 90, minimum=5, maximum=3600)
    )
    queue_admission_enabled = bool(queue_aware_admission and _env_flag("QUERYLAKE_SEARCH_QUEUE_ADMISSION_ENABLED", True))
    queue_throttle_enabled_resolved = bool(queue_throttle_enabled and _env_flag("QUERYLAKE_SEARCH_QUEUE_THROTTLE_ENABLED", True))
    queue_concurrency_current: Optional[int] = None
    queue_concurrency_remaining: Optional[int] = None
    queue_utilization: Optional[float] = None
    queue_throttle_applied = False
    queue_throttle_regime = "none"
    route_concurrency_key: Optional[str] = None
    route_concurrency_token: Optional[str] = None
    route_concurrency_released = False

    def _release_route_concurrency() -> None:
        nonlocal route_concurrency_released
        if route_concurrency_released:
            return
        if isinstance(route_concurrency_key, str) and isinstance(route_concurrency_token, str):
            release_concurrency(route_concurrency_key, route_concurrency_token)
        route_concurrency_released = True

    if adaptive_lane_routing:
        adaptive_profile_selected = adaptive_profile_selected or _infer_adaptive_query_profile(str(query.get("bm25", "")))
        resolved_policy = _resolve_adaptive_lane_policy(adaptive_lane_policy)
        adaptive_policy_source = "override" if isinstance(adaptive_lane_policy, dict) else "default"
        profile_policy = resolved_policy.get(adaptive_profile_selected) or resolved_policy.get(ADAPTIVE_PROFILE_DEFAULT)
        if profile_policy is not None:
            lane_limits = profile_policy.get("limits", {})
            lane_weights = profile_policy.get("weights", {})
            limit_bm25 = int(max(0, min(200, int(lane_limits.get("bm25", limit_bm25)))))
            limit_similarity = int(max(0, min(200, int(lane_limits.get("dense", limit_similarity)))))
            limit_sparse = int(max(0, min(200, int(lane_limits.get("sparse", limit_sparse)))))
            bm25_weight = float(max(0.0, float(lane_weights.get("bm25", bm25_weight))))
            similarity_weight = float(max(0.0, float(lane_weights.get("dense", similarity_weight))))
            sparse_weight = float(max(0.0, float(lane_weights.get("sparse", sparse_weight))))
            adaptive_lane_applied = True

    use_bm25 = bool(limit_bm25 > 0 and bm25_weight > 0) if explicit_use_bm25 is None else bool(explicit_use_bm25)
    use_similarity = bool(limit_similarity > 0 and similarity_weight > 0) if explicit_use_similarity is None else bool(explicit_use_similarity)
    use_sparse = bool(limit_sparse > 0 and sparse_weight > 0) if explicit_use_sparse is None else bool(explicit_use_sparse)
    configured_sparse_dims = configured_sparse_index_dimensions()
    if bool(use_sparse) and int(sparse_dimensions) != int(configured_sparse_dims):
        raise ValueError(
            "Sparse dimension mismatch: requested sparse_dimensions does not match configured sparse index dimensions "
            f"(requested={int(sparse_dimensions)}, configured={int(configured_sparse_dims)}, env=QUERYLAKE_SPARSE_INDEX_DIMENSIONS)."
        )

    assert isinstance(use_bm25, bool) and isinstance(use_similarity, bool) and isinstance(use_sparse, bool), \
        "`use_bm25`, `use_similarity`, and `use_sparse` must be booleans"

    if not use_bm25:
        limit_bm25 = 0
        bm25_weight = 0.0
    if not use_similarity:
        limit_similarity = 0
        similarity_weight = 0.0
    if not use_sparse:
        limit_sparse = 0
        sparse_weight = 0.0

    if dynamic_lane_budgeting:
        dynamic_budget_cap = int(dynamic_lane_total_limit_cap or _default_dynamic_limit_cap(adaptive_profile_selected))
        limit_bm25, limit_similarity, limit_sparse, dynamic_budget_applied = _apply_dynamic_lane_limit_schedule(
            limit_bm25=int(limit_bm25),
            limit_similarity=int(limit_similarity),
            limit_sparse=int(limit_sparse),
            bm25_weight=float(bm25_weight),
            similarity_weight=float(similarity_weight),
            sparse_weight=float(sparse_weight),
            use_bm25=bool(use_bm25),
            use_similarity=bool(use_similarity),
            use_sparse=bool(use_sparse),
            total_cap=int(dynamic_budget_cap),
            min_per_enabled=int(dynamic_lane_min_per_enabled),
        )

    assert (not use_bm25) or limit_bm25 > 0, "BM25 lane enabled but `limit_bm25` is 0"
    assert (not use_similarity) or limit_similarity > 0, "Dense lane enabled but `limit_similarity` is 0"
    assert (not use_sparse) or limit_sparse > 0, "Sparse lane enabled but `limit_sparse` is 0"

    assert (bm25_weight + similarity_weight + sparse_weight) > 0, \
        "At least one enabled lane must have positive weight"

    # Prevent SQL injection with dense embedding.
    if embedding is not None:
        assert len(embedding) == 1024 and all(isinstance(x, (int, float)) for x in embedding), \
            "Embedding must be a list of 1024 floats"
    t_2 = time.time()
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    assert_collections_priviledge(database, auth, collection_ids)
    t_3 = time.time()

    if web_search:
        collection_ids.append("WEB")
    
    if (not _direct_stage_call and not return_statement):
        resolved_pipeline_id = "orchestrated.search_hybrid"
        resolved_pipeline_version = "v1"
        pipeline_resolution = {"source": "fallback", "notes": ["pre_resolution_default"]}
        try:
            if queue_admission_enabled and queue_admission_limit_resolved > 0:
                route_concurrency_key = "ql:rl:concurrency:search_hybrid"
                route_concurrency_token = get_request_id() or f"search_{uuid.uuid4().hex}"
                lease = acquire_concurrency(
                    route_concurrency_key,
                    limit=int(queue_admission_limit_resolved),
                    ttl_seconds=int(queue_admission_ttl_resolved),
                    token=str(route_concurrency_token),
                )
                queue_admission_applied = True
                queue_concurrency_current = int(lease.current) if lease.current is not None else None
                queue_concurrency_remaining = int(lease.remaining) if lease.remaining is not None else None
                if queue_concurrency_current is not None and queue_admission_limit_resolved > 0:
                    queue_utilization = min(2.0, max(0.0, float(queue_concurrency_current) / float(queue_admission_limit_resolved)))
                elif queue_concurrency_remaining is not None and queue_admission_limit_resolved > 0:
                    queue_utilization = min(
                        2.0,
                        max(
                            0.0,
                            float(queue_admission_limit_resolved - queue_concurrency_remaining) / float(queue_admission_limit_resolved),
                        ),
                    )
                if not bool(lease.allowed):
                    metrics.rate_limit_denied("search_hybrid")
                    raise RuntimeError(
                        f"Search concurrency limit reached (limit={int(queue_admission_limit_resolved)}, "
                        f"retry_after={int(lease.retry_after_seconds or queue_admission_ttl_resolved)}s)"
                    )

            if queue_throttle_enabled_resolved:
                limit_bm25, limit_similarity, limit_sparse, queue_throttle_applied, queue_throttle_regime = _apply_queue_pressure_schedule(
                    queue_utilization=queue_utilization,
                    limit_bm25=int(limit_bm25),
                    limit_similarity=int(limit_similarity),
                    limit_sparse=int(limit_sparse),
                    bm25_weight=float(bm25_weight),
                    similarity_weight=float(similarity_weight),
                    sparse_weight=float(sparse_weight),
                    use_bm25=bool(use_bm25),
                    use_similarity=bool(use_similarity),
                    use_sparse=bool(use_sparse),
                    min_per_enabled=int(dynamic_lane_min_per_enabled),
                    soft_utilization=float(queue_throttle_soft_utilization),
                    hard_utilization=float(queue_throttle_hard_utilization),
                    soft_scale=float(queue_throttle_soft_scale),
                    hard_scale=float(queue_throttle_hard_scale),
                    disable_sparse_at_hard=bool(queue_throttle_disable_sparse_at_hard),
                )
                use_bm25 = bool(limit_bm25 > 0 and bm25_weight > 0)
                use_similarity = bool(limit_similarity > 0 and similarity_weight > 0)
                use_sparse = bool(limit_sparse > 0 and sparse_weight > 0)

            fallback_pipeline = default_pipeline_for_route("search_hybrid")
            pipeline, pipeline_resolution = _resolve_route_pipeline(
                database,
                route="search_hybrid",
                user_name=user_auth.username,
                auth=auth,
                pipeline_override=_pipeline_override,
                fallback_pipeline=fallback_pipeline,
            )
            resolved_pipeline_id = pipeline.pipeline_id
            resolved_pipeline_version = pipeline.version

            if use_sparse and limit_sparse > 0:
                has_sparse_stage = any(
                    str(stage.stage_id) == "sparse" or str(stage.primitive_id) == "SparseRetrieverPGVector"
                    for stage in pipeline.stages
                )
                if not has_sparse_stage:
                    augmented = pipeline.model_copy(deep=True)
                    augmented.stages.append(
                        RetrievalPipelineStage(
                            stage_id="sparse",
                            primitive_id="SparseRetrieverPGVector",
                            config={
                                "group_chunks": False,
                                "limit_key": "limit_sparse",
                                "sparse_query_text_key": "sparse_query_text",
                                "sparse_query_value_key": "sparse_query_value",
                                "sparse_embedding_function": "embedding_sparse",
                                "sparse_dimensions": int(sparse_dimensions),
                            },
                            enabled=True,
                        )
                    )
                    pipeline = augmented

            orchestrated_query_embedding = embedding
            if use_similarity:
                assert "embedding" in query or orchestrated_query_embedding is not None, \
                    "If similarity lane is enabled, `query['embedding']` or `embedding` must be provided"
                if orchestrated_query_embedding is None:
                    embedding_key = build_embedding_cache_key(user_auth.username, str(query.get("embedding", "")))
                    embedding_cached = get_embedding(embedding_key)
                    if embedding_cached is not None:
                        orchestrated_query_embedding = embedding_cached
                    else:
                        embedding_call: Awaitable[Callable] = toolchain_function_caller("embedding")
                        embedding_payload = (await embedding_call(auth, [str(query.get("embedding", ""))]))[0]
                        orchestrated_query_embedding = _extract_dense_embedding_payload(embedding_payload)
                        assert orchestrated_query_embedding is not None and len(orchestrated_query_embedding) == 1024, \
                            "Dense embedding function did not return a valid 1024-d vector"
                        set_embedding(embedding_key, orchestrated_query_embedding)

            orchestrated_sparse_query_value = embedding_sparse
            if use_sparse:
                sparse_value = (
                    _normalize_sparse_query_value(
                        orchestrated_sparse_query_value,
                        sparse_dimensions,
                        strict_dimensions=True,
                        source_label="embedding_sparse_input",
                    )
                    if orchestrated_sparse_query_value is not None
                    else None
                )
                if sparse_value is None:
                    sparse_value = _normalize_sparse_query_value(
                        query.get("sparse"),
                        sparse_dimensions,
                        strict_dimensions=True,
                        source_label="query.sparse",
                    )
                if sparse_value is None:
                    sparse_query_text = str(query.get("sparse", query.get("bm25", "")))
                    sparse_cache_key = build_embedding_cache_key(user_auth.username, f"sparse::{sparse_query_text}")
                    sparse_cached = get_embedding(sparse_cache_key)
                    if sparse_cached is not None:
                        sparse_value = _normalize_sparse_query_value(
                            sparse_cached,
                            sparse_dimensions,
                            strict_dimensions=True,
                            source_label="sparse_cache",
                        )
                    if sparse_value is None:
                        sparse_embedding_call: Optional[Callable[..., Awaitable[Any]]] = None
                        try:
                            sparse_embedding_call = toolchain_function_caller(sparse_embedding_function)
                        except Exception:
                            if sparse_embedding_function != "embedding":
                                sparse_embedding_call = toolchain_function_caller("embedding")
                        if sparse_embedding_call is not None:
                            sparse_payload = (await sparse_embedding_call(auth, [sparse_query_text]))[0]
                            sparse_value = _normalize_sparse_query_value(
                                sparse_payload,
                                sparse_dimensions,
                                strict_dimensions=True,
                                source_label=f"{sparse_embedding_function}[0]",
                            )
                            if sparse_value is not None:
                                set_embedding(sparse_cache_key, sparse_value.to_text())
                assert sparse_value is not None, \
                    "Sparse lane is enabled but sparse embedding could not be resolved from input/query"
                orchestrated_sparse_query_value = sparse_value

            request_obj = RetrievalRequest(
                route="search_hybrid",
                query_text=str(query.get("bm25", "")),
                query_embedding=orchestrated_query_embedding,
                collection_ids=collection_ids,
                options={
                    "limit": int(limit_bm25 + limit_similarity + limit_sparse),
                    "limit_bm25": int(limit_bm25),
                    "offset_bm25": 0,
                    "limit_similarity": int(limit_similarity),
                    "limit_sparse": int(limit_sparse),
                    "bm25_query_text": str(query.get("bm25", "")),
                    "bm25_catch_all_fields": list(resolved_catch_all_fields),
                    "dense_query_text": str(query.get("embedding", query.get("bm25", ""))),
                    "sparse_query_text": str(query.get("sparse", query.get("bm25", ""))),
                    "sparse_query_value": orchestrated_sparse_query_value,
                    "sparse_embedding_function": str(sparse_embedding_function),
                    "sparse_dimensions": int(sparse_dimensions),
                    "sparse_prune_max_terms": sparse_prune_max_terms,
                    "sparse_prune_min_abs_weight": float(sparse_prune_min_abs_weight),
                    "sparse_calibration": str(sparse_calibration),
                    "use_bm25": bool(use_bm25),
                    "use_similarity": bool(use_similarity),
                    "use_sparse": bool(use_sparse),
                    "adaptive_lane_routing": bool(adaptive_lane_routing),
                    "adaptive_lane_applied": bool(adaptive_lane_applied),
                    "adaptive_query_profile": str(adaptive_profile_selected),
                    "adaptive_lane_policy_source": str(adaptive_policy_source),
                    "adaptive_lane_policy": adaptive_lane_policy if isinstance(adaptive_lane_policy, dict) else None,
                    "dynamic_lane_budgeting": bool(dynamic_lane_budgeting),
                    "dynamic_lane_budget_applied": bool(dynamic_budget_applied),
                    "dynamic_lane_total_limit_cap": int(dynamic_budget_cap),
                    "dynamic_lane_min_per_enabled": int(dynamic_lane_min_per_enabled),
                    "queue_aware_admission": bool(queue_admission_enabled),
                    "queue_admission_applied": bool(queue_admission_applied),
                    "queue_admission_concurrency_limit": int(queue_admission_limit_resolved),
                    "queue_admission_ttl_seconds": int(queue_admission_ttl_resolved),
                    "queue_throttle_enabled": bool(queue_throttle_enabled_resolved),
                    "queue_throttle_applied": bool(queue_throttle_applied),
                    "queue_throttle_regime": str(queue_throttle_regime),
                    "queue_utilization": queue_utilization,
                    "queue_throttle_soft_utilization": float(queue_throttle_soft_utilization),
                    "queue_throttle_hard_utilization": float(queue_throttle_hard_utilization),
                    "queue_throttle_soft_scale": float(queue_throttle_soft_scale),
                    "queue_throttle_hard_scale": float(queue_throttle_hard_scale),
                    "queue_current": queue_concurrency_current,
                    "queue_remaining": queue_concurrency_remaining,
                    "strict_constraint_prefilter": bool(strict_constraint_prefilter),
                    "constraint_query_text": str(constraint_query_text) if isinstance(constraint_query_text, str) else None,
                    "fusion_primitive": str(fusion_primitive) if isinstance(fusion_primitive, str) and len(fusion_primitive.strip()) > 0 else None,
                    "fusion_normalization": str(fusion_normalization) if isinstance(fusion_normalization, str) and len(fusion_normalization.strip()) > 0 else "minmax",
                    "rrf_k": int(rrf_k),
                    "explain_plan": bool(explain_plan),
                    "fusion_weights": {"bm25": float(bm25_weight), "dense": float(similarity_weight), "sparse": float(sparse_weight)},
                    "fusion_score_keys": {"bm25": "bm25_score", "dense": "similarity_score", "sparse": "sparse_score"},
                    "rerank_enabled": bool("rerank" in query),
                    "rerank_query_text": str(query.get("rerank", query.get("bm25", ""))),
                    "web_search": bool(web_search),
                    "_skip_observability": True,
                },
            )

            async def _direct_hybrid_stage(**kwargs):
                direct_kwargs = dict(kwargs)
                direct_kwargs["_direct_stage_call"] = True
                direct_kwargs["_skip_observability"] = True
                return await search_hybrid(**direct_kwargs)

            retrievers = _build_retrievers_for_pipeline(
                pipeline=pipeline,
                database=database,
                auth=auth,
                toolchain_function_caller=toolchain_function_caller,
                search_bm25_fn=search_bm25,
                search_hybrid_fn=_direct_hybrid_stage,
            )
            fusion_instance = _select_fusion_for_pipeline(pipeline=pipeline, options=request_obj.options)
            reranker_primitive = _select_reranker_for_pipeline(
                pipeline=pipeline,
                auth=auth,
                toolchain_function_caller=toolchain_function_caller,
                options=request_obj.options,
            )
            packer = _select_packer(
                group_chunks=group_chunks,
                options={**request_obj.options, **({"packing_mode": pipeline.flags.get("packing_mode")} if isinstance(pipeline.flags, dict) and pipeline.flags.get("packing_mode") is not None else {})},
            )

            run_result = await PipelineOrchestrator().run(
                request=request_obj,
                pipeline=pipeline,
                retrievers=retrievers,
                fusion=fusion_instance,
                reranker=reranker_primitive,
                packer=packer,
            )
            rows_as_chunks = [_candidate_to_document_chunk(candidate) for candidate in run_result.candidates]
            results_dump = []
            for idx, row in enumerate(rows_as_chunks):
                dumped = row.model_dump()
                rerank_score = run_result.candidates[idx].stage_scores.get("rerank_score")
                if rerank_score is not None:
                    dumped["rerank_score"] = float(rerank_score)
                results_dump.append(dumped)

            duration_map = {trace.stage: trace.duration_ms / 1000.0 for trace in run_result.traces}
            duration_map["total"] = time.time() - t_1
            result_metrics = {"duration": duration_map, "cache": {}}
            plan_explain_payload = None
            if explain_plan:
                plan_explain_payload = build_retrieval_plan_explain(
                    route="search_hybrid",
                    pipeline=pipeline,
                    options=request_obj.options,
                    pipeline_resolution=pipeline_resolution,
                    lane_state={
                        "use_bm25": bool(use_bm25),
                        "use_similarity": bool(use_similarity),
                        "use_sparse": bool(use_sparse),
                        "strict_constraint_prefilter": bool(strict_constraint_prefilter),
                        "adaptive_lane_routing": bool(adaptive_lane_routing),
                        "adaptive_lane_applied": bool(adaptive_lane_applied),
                        "adaptive_query_profile": str(adaptive_profile_selected),
                        "dynamic_lane_budgeting": bool(dynamic_lane_budgeting),
                        "dynamic_lane_budget_applied": bool(dynamic_budget_applied),
                        "dynamic_lane_total_limit_cap": int(dynamic_budget_cap),
                        "queue_aware_admission": bool(queue_admission_enabled),
                        "queue_admission_applied": bool(queue_admission_applied),
                        "queue_admission_concurrency_limit": int(queue_admission_limit_resolved),
                        "queue_admission_ttl_seconds": int(queue_admission_ttl_resolved),
                        "queue_throttle_enabled": bool(queue_throttle_enabled_resolved),
                        "queue_throttle_applied": bool(queue_throttle_applied),
                        "queue_throttle_regime": str(queue_throttle_regime),
                        "queue_utilization": queue_utilization,
                        "queue_current": queue_concurrency_current,
                        "queue_remaining": queue_concurrency_remaining,
                        "bm25_catch_all_fields": list(resolved_catch_all_fields),
                    },
                )

            if not _skip_observability:
                metrics.record_retrieval(
                    route="search_hybrid",
                    status="ok",
                    latency_seconds=result_metrics.get("duration", {}).get("total", 0.0),
                    results_count=len(results_dump),
                )
                log_retrieval_run(
                    database,
                    route="search_hybrid",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=collection_ids,
                    pipeline_id=run_result.pipeline_id,
                    pipeline_version=run_result.pipeline_version,
                    filters={"collection_ids": collection_ids, "web_search": web_search, "orchestrated": True},
                    budgets={
                        "limit_bm25": limit_bm25,
                        "limit_similarity": limit_similarity,
                        "limit_sparse": limit_sparse,
                        "rerank_enabled": bool("rerank" in query),
                        "queue_admission_applied": bool(queue_admission_applied),
                        "queue_admission_concurrency_limit": int(queue_admission_limit_resolved),
                        "queue_admission_ttl_seconds": int(queue_admission_ttl_resolved),
                        "queue_throttle_applied": bool(queue_throttle_applied),
                        "queue_throttle_regime": str(queue_throttle_regime),
                    },
                    timings=result_metrics.get("duration", {}),
                    counters={
                        "rows_returned": len(results_dump),
                        "bm25_weight": float(bm25_weight),
                        "similarity_weight": float(similarity_weight),
                        "sparse_weight": float(sparse_weight),
                        "sparse_dimensions": int(sparse_dimensions),
                        "use_bm25": bool(use_bm25),
                        "use_similarity": bool(use_similarity),
                        "use_sparse": bool(use_sparse),
                        "adaptive_lane_routing": bool(adaptive_lane_routing),
                        "adaptive_lane_applied": bool(adaptive_lane_applied),
                        "adaptive_query_profile": str(adaptive_profile_selected),
                        "adaptive_lane_policy_source": str(adaptive_policy_source),
                        "dynamic_lane_budgeting": bool(dynamic_lane_budgeting),
                        "dynamic_lane_budget_applied": bool(dynamic_budget_applied),
                        "dynamic_lane_total_limit_cap": int(dynamic_budget_cap),
                        "dynamic_lane_min_per_enabled": int(dynamic_lane_min_per_enabled),
                        "queue_aware_admission": bool(queue_admission_enabled),
                        "queue_admission_applied": bool(queue_admission_applied),
                        "queue_admission_concurrency_limit": int(queue_admission_limit_resolved),
                        "queue_admission_ttl_seconds": int(queue_admission_ttl_resolved),
                        "queue_throttle_enabled": bool(queue_throttle_enabled_resolved),
                        "queue_throttle_applied": bool(queue_throttle_applied),
                        "queue_throttle_regime": str(queue_throttle_regime),
                        "queue_utilization": queue_utilization,
                        "queue_current": queue_concurrency_current,
                        "queue_remaining": queue_concurrency_remaining,
                        "strict_constraint_prefilter": bool(strict_constraint_prefilter),
                        "fusion_primitive": str(request_obj.options.get("fusion_primitive") or (pipeline.flags or {}).get("fusion_primitive", "WeightedScoreFusion")),
                        "fusion_normalization": str(request_obj.options.get("fusion_normalization", "minmax")),
                        "rrf_k": int(rrf_k),
                        "group_chunks": bool(group_chunks),
                        "bm25_catch_all_fields": list(resolved_catch_all_fields),
                    },
                    result_rows=results_dump,
                    candidate_details=[candidate.model_dump() for candidate in run_result.candidates],
                    status="ok",
                    md={
                        "stage_trace": [trace.model_dump() for trace in run_result.traces],
                        "pipeline_resolution": pipeline_resolution,
                        **({"plan_explain": plan_explain_payload} if plan_explain_payload is not None else {}),
                    },
                )
            database.rollback()
            response_payload = {"rows": results_dump, **result_metrics}
            if plan_explain_payload is not None:
                response_payload["plan_explain"] = plan_explain_payload
            if queue_admission_applied or queue_throttle_applied:
                response_payload["queue_control"] = {
                    "admission_applied": bool(queue_admission_applied),
                    "admission_limit": int(queue_admission_limit_resolved),
                    "admission_ttl_seconds": int(queue_admission_ttl_resolved),
                    "queue_utilization": queue_utilization,
                    "queue_current": queue_concurrency_current,
                    "queue_remaining": queue_concurrency_remaining,
                    "throttle_applied": bool(queue_throttle_applied),
                    "throttle_regime": str(queue_throttle_regime),
                    "limits": {
                        "limit_bm25": int(limit_bm25),
                        "limit_similarity": int(limit_similarity),
                        "limit_sparse": int(limit_sparse),
                    },
                }
            _release_route_concurrency()
            return response_payload
        except Exception as e:
            _release_route_concurrency()
            if not _skip_observability:
                metrics.record_retrieval(
                    route="search_hybrid",
                    status="error",
                    latency_seconds=time.time() - t_1,
                    results_count=0,
                )
                log_retrieval_run(
                    database,
                    route="search_hybrid",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=collection_ids,
                    pipeline_id=resolved_pipeline_id,
                    pipeline_version=resolved_pipeline_version,
                    filters={"collection_ids": collection_ids, "web_search": web_search, "orchestrated": True},
                    budgets={
                        "limit_bm25": limit_bm25,
                        "limit_similarity": limit_similarity,
                        "limit_sparse": limit_sparse,
                        "queue_admission_applied": bool(queue_admission_applied),
                        "queue_admission_concurrency_limit": int(queue_admission_limit_resolved),
                        "queue_admission_ttl_seconds": int(queue_admission_ttl_resolved),
                        "queue_throttle_applied": bool(queue_throttle_applied),
                        "queue_throttle_regime": str(queue_throttle_regime),
                    },
                    timings={"total": time.time() - t_1},
                    counters={},
                    result_rows=[],
                    status="error",
                    error=str(e),
                    md={"pipeline_resolution": pipeline_resolution},
                )
            database.rollback()
            raise e
    
    embedding_cache_hit = False
    sparse_embedding_cache_hit = False
    rerank_cache_hits = 0
    rerank_cache_misses = 0
    sparse_terms_before = 0
    sparse_terms_after = 0

    if use_similarity:
        assert "embedding" in query or embedding is not None, \
            "If similarity lane is enabled, `query['embedding']` or `embedding` must be provided"
        if embedding is None:
            embedding_key = build_embedding_cache_key(user_auth.username, str(query.get("embedding", "")))
            embedding_cached = get_embedding(embedding_key)
            if embedding_cached is not None:
                embedding = embedding_cached
                embedding_cache_hit = True
            else:
                embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
                embedding_payload = (await embedding_call(auth, [str(query.get("embedding", ""))]))[0]
                embedding = _extract_dense_embedding_payload(embedding_payload)
                assert embedding is not None and len(embedding) == 1024, \
                    "Dense embedding function did not return a valid 1024-d vector"
                set_embedding(embedding_key, embedding)
    else:
        embedding = [0.0] * 1024

    if use_sparse:
        sparse_value = (
            _normalize_sparse_query_value(
                embedding_sparse,
                sparse_dimensions,
                strict_dimensions=True,
                source_label="embedding_sparse_input",
            )
            if embedding_sparse is not None
            else None
        )
        if sparse_value is None:
            sparse_value = _normalize_sparse_query_value(
                query.get("sparse"),
                sparse_dimensions,
                strict_dimensions=True,
                source_label="query.sparse",
            )
        if sparse_value is None:
            sparse_query_text = str(query.get("sparse", query.get("bm25", "")))
            sparse_cache_key = build_embedding_cache_key(user_auth.username, f"sparse::{sparse_query_text}")
            sparse_cached = get_embedding(sparse_cache_key)
            if sparse_cached is not None:
                sparse_value = _normalize_sparse_query_value(
                    sparse_cached,
                    sparse_dimensions,
                    strict_dimensions=True,
                    source_label="sparse_cache",
                )
                sparse_embedding_cache_hit = sparse_value is not None
            if sparse_value is None:
                sparse_embedding_call: Optional[Callable[..., Awaitable[Any]]] = None
                try:
                    sparse_embedding_call = toolchain_function_caller(sparse_embedding_function)
                except Exception:
                    if sparse_embedding_function != "embedding":
                        sparse_embedding_call = toolchain_function_caller("embedding")
                if sparse_embedding_call is not None:
                    sparse_payload = (await sparse_embedding_call(auth, [sparse_query_text]))[0]
                    sparse_value = _normalize_sparse_query_value(
                        sparse_payload,
                        sparse_dimensions,
                        strict_dimensions=True,
                        source_label=f"{sparse_embedding_function}[0]",
                    )
                    if sparse_value is not None:
                        set_embedding(sparse_cache_key, sparse_value.to_text())
        assert sparse_value is not None, \
            "Sparse lane is enabled but sparse embedding could not be resolved from input/query"
        sparse_value, sparse_terms_before, sparse_terms_after = _apply_sparse_pruning_and_calibration(
            sparse_value,
            max_terms=sparse_prune_max_terms,
            min_abs_weight=float(sparse_prune_min_abs_weight),
            calibration=sparse_calibration,
        )
        embedding_sparse = sparse_value
    else:
        embedding_sparse = pgvector.SparseVector({}, sparse_dimensions)
    t_4 = time.time()

    bm25_query_text = str(query.get("bm25", ""))
    formatted_query, strong_where_clause = parse_search(
        bm25_query_text,
        CHUNK_INDEXED_COLUMNS,
        catch_all_fields=resolved_catch_all_fields,
    )
    if strict_constraint_prefilter:
        constraint_seed = (
            str(constraint_query_text)
            if isinstance(constraint_query_text, str) and len(str(constraint_query_text).strip()) > 0
            else bm25_query_text
        )
        _, strict_clause = parse_search(
            constraint_seed,
            CHUNK_INDEXED_COLUMNS,
            catch_all_fields=resolved_catch_all_fields,
        )
        if isinstance(strict_clause, str) and len(strict_clause.strip()) > 0:
            strong_where_clause = strict_clause
    if use_bm25 and str(formatted_query).strip() == "()":
        use_bm25 = False
        limit_bm25 = 0
        bm25_weight = 0.0
        if (similarity_weight + sparse_weight) <= 0:
            raise AssertionError("BM25 query is empty and no other lane has positive weight")

    collection_spec = f"""collection_id:IN {str(collection_ids).replace("'", "")}"""
    collection_spec_new = f"""collection_id IN ({str(collection_ids)[1:-1]})"""
    similarity_constraint = (
        f"WHERE id @@@ paradedb.parse('({collection_spec}) AND ({strong_where_clause})')"
        if strong_where_clause is not None
        else f"WHERE {collection_spec_new}"
    )
    t_5 = time.time()

    bm25_filter_expr = f"({collection_spec}) AND ({formatted_query})"
    if strict_constraint_prefilter and strong_where_clause is not None:
        bm25_filter_expr = f"({collection_spec}) AND ({strong_where_clause}) AND ({formatted_query})"

    bm25_ranked_cte = f"""
    bm25_candidates AS (
        SELECT id
        FROM {DocumentChunk.__tablename__}
        WHERE id @@@ paradedb.parse('{bm25_filter_expr}')
        ORDER BY paradedb.score(id) DESC
        LIMIT :limit_bm25
    ),
    bm25_ranked AS (
        SELECT id, RANK() OVER (ORDER BY paradedb.score(id) DESC) AS rank
        FROM bm25_candidates
    )
    """ if use_bm25 else """
    bm25_ranked AS (
        SELECT NULL::text AS id, NULL::bigint AS rank WHERE FALSE
    )
    """

    semantic_search_cte = f"""
    semantic_search AS (
        SELECT id, RANK() OVER (ORDER BY embedding <=> :embedding_in) AS rank
        FROM {DocumentChunk.__tablename__}
        {similarity_constraint} AND embedding IS NOT NULL
        ORDER BY embedding <=> :embedding_in
        LIMIT :limit_similarity
    )
    """ if use_similarity else """
    semantic_search AS (
        SELECT NULL::text AS id, NULL::bigint AS rank WHERE FALSE
    )
    """

    sparse_search_cte = f"""
    sparse_search AS (
        SELECT id, RANK() OVER (
            ORDER BY (embedding_sparse::sparsevec({int(sparse_dimensions)})) <=> CAST(:sparse_embedding_in AS sparsevec({int(sparse_dimensions)}))
        ) AS rank
        FROM {DocumentChunk.__tablename__}
        {similarity_constraint} AND embedding_sparse IS NOT NULL
        ORDER BY (embedding_sparse::sparsevec({int(sparse_dimensions)})) <=> CAST(:sparse_embedding_in AS sparsevec({int(sparse_dimensions)}))
        LIMIT :limit_sparse
    )
    """ if use_sparse else """
    sparse_search AS (
        SELECT NULL::text AS id, NULL::bigint AS rank WHERE FALSE
    )
    """

    total_limit = int(limit_bm25 + limit_similarity + limit_sparse)
    assert total_limit > 0, "At least one lane must have a positive limit"

    stmt_text = text(f"""
        WITH
        {bm25_ranked_cte},
        {semantic_search_cte},
        {sparse_search_cte},
        candidate_ids AS (
            SELECT id FROM bm25_ranked
            UNION
            SELECT id FROM semantic_search
            UNION
            SELECT id FROM sparse_search
        )
        SELECT
            candidate_ids.id AS id,
            COALESCE(1.0 / (60 + semantic_search.rank), 0.0) AS semantic_score,
            COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) AS bm25_score,
            COALESCE(1.0 / (60 + sparse_search.rank), 0.0) AS sparse_score,
            (
                COALESCE(1.0 / (60 + semantic_search.rank), 0.0) * :similarity_weight +
                COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) * :bm25_weight +
                COALESCE(1.0 / (60 + sparse_search.rank), 0.0) * :sparse_weight
            ) AS score,
            {retrieved_fields_string}
        FROM candidate_ids
        LEFT JOIN semantic_search ON candidate_ids.id = semantic_search.id
        LEFT JOIN bm25_ranked ON candidate_ids.id = bm25_ranked.id
        LEFT JOIN sparse_search ON candidate_ids.id = sparse_search.id
        JOIN {DocumentChunk.__tablename__} ON {DocumentChunk.__tablename__}.id = candidate_ids.id
        ORDER BY score DESC, text
        LIMIT :sum;
    """)
    bind_values = {
        "similarity_weight": float(similarity_weight),
        "bm25_weight": float(bm25_weight),
        "sparse_weight": float(sparse_weight),
        "sum": total_limit,
    }
    if use_bm25:
        bind_values["limit_bm25"] = limit_bm25
    if use_similarity:
        bind_values["limit_similarity"] = limit_similarity
        bind_values["embedding_in"] = str(embedding)
    if use_sparse:
        bind_values["limit_sparse"] = limit_sparse
        bind_values["sparse_embedding_in"] = (
            embedding_sparse.to_text()
            if hasattr(embedding_sparse, "to_text")
            else str(embedding_sparse)
        )
    STMT = stmt_text.bindparams(**bind_values)
    
    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))

    try:
        results = database.exec(STMT)
        results = list(results)
        results = list(filter(lambda x: not x[0] is None, results))
        t_6 = time.time()
        
        
        result_metrics = {
            "duration": {
                "input_assertions": t_2 - t_1,
                "collection_permission_assertions": t_3 - t_2,
                "embedding_call": t_4 - t_3,
                "query_parsing": t_5 - t_4,
                "query_execution": t_6 - t_5,
                "total": t_6 - t_1
            }
        }
        
        # results = list(filter(lambda x: not x[0] in id_exclusions, results))
        
        results_made : List[DocumentChunkDictionary] = list(map(lambda x: convert_chunk_query_result(x[5:]), results))
        
        for i, chunk in enumerate(results):
            results_made[i].similarity_score = float(chunk[1])
            results_made[i].bm25_score = float(chunk[2])
            results_made[i].sparse_score = float(chunk[3])
            results_made[i].hybrid_score = float(chunk[4])
        
        
        if group_chunks:
            results_made = group_adjacent_chunks(results_made)
        
        results = sorted(results_made, key=lambda x: x.hybrid_score, reverse=True)
        
        if "rerank" in query:
            rerank_call : Awaitable[Callable] = toolchain_function_caller("rerank")

            rerank_scores = [None for _ in results]
            rerank_batch = []
            rerank_batch_indices = []
            rerank_batch_keys = []

            for i, doc in enumerate(results):
                if len(doc.text) == 0:
                    rerank_scores[i] = 0.0
                    continue
                cache_key = build_rerank_cache_key(user_auth.username, query["rerank"], doc.text)
                cached_score = get_rerank_score(cache_key)
                if cached_score is None:
                    rerank_cache_misses += 1
                    rerank_batch_indices.append(i)
                    rerank_batch_keys.append(cache_key)
                    rerank_batch.append((query["rerank"], doc.text))
                else:
                    rerank_cache_hits += 1
                    rerank_scores[i] = float(cached_score)

            if len(rerank_batch) > 0:
                rerank_scores_new = await rerank_call(auth, rerank_batch)
                for i, score in enumerate(rerank_scores_new):
                    row_i = rerank_batch_indices[i]
                    rerank_scores[row_i] = float(score)
                    set_rerank_score(rerank_batch_keys[i], float(score))
            
            results : List[DocumentChunkDictionaryReranked] = list(map(lambda x: DocumentChunkDictionaryReranked(
                **results[x].model_dump(), 
                rerank_score=rerank_scores[x]
            ), list(range(len(results)))))
            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            t_7 = time.time()
            result_metrics["duration"].update({"rerank": t_7 - t_6, "total": t_7 - t_1})

        result_metrics["cache"] = {
            "embedding_hit": embedding_cache_hit,
            "sparse_embedding_hit": sparse_embedding_cache_hit,
            "rerank_hits": rerank_cache_hits,
            "rerank_misses": rerank_cache_misses,
        }
        
        results = [r.model_dump() for r in results]
        if not _skip_observability:
            metrics.record_retrieval(
                route="search_hybrid",
                status="ok",
                latency_seconds=result_metrics.get("duration", {}).get("total", 0.0),
                results_count=len(results),
            )
            log_retrieval_run(
                database,
                route="search_hybrid",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=collection_ids,
                pipeline_id="orchestrated.search_hybrid.sql",
                pipeline_version="v1",
                filters={"collection_ids": collection_ids, "web_search": web_search},
                budgets={
                    "limit_bm25": limit_bm25,
                    "limit_similarity": limit_similarity,
                    "limit_sparse": limit_sparse,
                    "rerank_enabled": bool("rerank" in query),
                },
                timings=result_metrics.get("duration", {}),
                counters={
                    "rows_returned": len(results),
                    "bm25_weight": float(bm25_weight),
                    "similarity_weight": float(similarity_weight),
                    "sparse_weight": float(sparse_weight),
                    "sparse_dimensions": int(sparse_dimensions),
                    "sparse_terms_before": int(sparse_terms_before),
                    "sparse_terms_after": int(sparse_terms_after),
                    "sparse_prune_max_terms": sparse_prune_max_terms,
                    "sparse_prune_min_abs_weight": float(sparse_prune_min_abs_weight),
                    "sparse_calibration": str(sparse_calibration),
                    "use_bm25": bool(use_bm25),
                    "use_similarity": bool(use_similarity),
                    "use_sparse": bool(use_sparse),
                    "adaptive_lane_routing": bool(adaptive_lane_routing),
                    "adaptive_lane_applied": bool(adaptive_lane_applied),
                    "adaptive_query_profile": str(adaptive_profile_selected),
                    "adaptive_lane_policy_source": str(adaptive_policy_source),
                    "group_chunks": bool(group_chunks),
                    "bm25_catch_all_fields": list(resolved_catch_all_fields),
                    "embedding_cache_hit": embedding_cache_hit,
                    "sparse_embedding_cache_hit": sparse_embedding_cache_hit,
                    "rerank_cache_hits": rerank_cache_hits,
                    "rerank_cache_misses": rerank_cache_misses,
                },
                result_rows=results,
                status="ok",
            )
        response_payload = {"rows": results, **result_metrics}
        if explain_plan:
            response_payload["plan_explain"] = {
                "route": "search_hybrid",
                "pipeline": {
                    "pipeline_id": "orchestrated.search_hybrid.sql",
                    "pipeline_version": "v1",
                    "source": "legacy_sql",
                    "stages": [
                        {"stage_id": "bm25", "enabled": bool(use_bm25)},
                        {"stage_id": "dense", "enabled": bool(use_similarity)},
                        {"stage_id": "sparse", "enabled": bool(use_sparse)},
                    ],
                },
                "effective": {
                    "fusion": {
                        "enabled": True,
                        "primitive": "WeightedScoreFusion",
                        "normalization": "rrf-weighted-sql",
                    },
                    "limits": {
                        "limit_bm25": int(limit_bm25),
                        "limit_similarity": int(limit_similarity),
                        "limit_sparse": int(limit_sparse),
                    },
                },
            }
        database.rollback()
        return response_payload
        
    except Exception as e:
        if not _skip_observability:
            metrics.record_retrieval(
                route="search_hybrid",
                status="error",
                latency_seconds=time.time() - t_1,
                results_count=0,
            )
            log_retrieval_run(
                database,
                route="search_hybrid",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=collection_ids,
                pipeline_id="orchestrated.search_hybrid.sql",
                pipeline_version="v1",
                filters={"collection_ids": collection_ids, "web_search": web_search},
                budgets={
                    "limit_bm25": limit_bm25,
                    "limit_similarity": limit_similarity,
                    "limit_sparse": limit_sparse,
                },
                timings={"total": time.time() - t_1},
                counters={},
                result_rows=[],
                status="error",
                error=str(e),
            )
        database.rollback()
        raise e

def search_bm25(
    database: Session,
    auth : AuthType,
    query: str,
    collection_ids: List[str] = [],
    limit: int = 10,
    offset: int = 0,
    web_search : bool = False,
    return_statement : bool = False,
    group_chunks : bool = True,
    table : Literal["document_chunk", "document", "segment"] = "document_chunk",
    sort_by: str = "score",
    sort_dir: Literal["DESC", "ASC"] = "DESC",
    _direct_stage_call: bool = False,
    _skip_observability: bool = False,
    _pipeline_override: Optional[Dict[str, str]] = None,
) -> List[DocumentChunkDictionary]:
    t_1 = time.time()
    
    (_, user_auth) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(limit, int) and limit >= 0 and limit <= 200), \
        "limit must be an int between 0 and 200"
    
    assert (isinstance(offset, int) and offset >= 0), \
        "offset must be an int greater than or equal to 0"
    
    assert isinstance(query, str), \
        "query must be a string"
    
    assert table in ["document_chunk", "document", "segment"], \
        "`table` must be either 'document_chunk', 'document', or 'segment'"

    if table == "segment":
        assert _env_flag("QUERYLAKE_RETRIEVAL_SEGMENT_ENABLED", False), \
            "segment retrieval is disabled; set QUERYLAKE_RETRIEVAL_SEGMENT_ENABLED=1 to enable it"
    
    group_chunks = group_chunks and (table == "document_chunk")
    
    valid_fields, chosen_table, chosen_attributes, chosen_catch_alls = {
        "document_chunk": (CHUNK_INDEXED_COLUMNS, DocumentChunk, retrieved_fields_string, ["text"]),
        "document": (DOCUMENT_INDEXED_COLUMNS, document_raw, retrieved_document_fields_string, ["file_name"]),
        "segment": (SEGMENT_INDEXED_COLUMNS, SegmentTable, retrieved_segment_fields_string, ["text"]),
    }[table]
    
    
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    assert_collections_priviledge(database, auth, collection_ids)

    if (
        not _direct_stage_call
        and not return_statement
        and table == "document_chunk"
    ):
        resolved_pipeline_id = "orchestrated.search_bm25.document_chunk"
        resolved_pipeline_version = "v1"
        pipeline_resolution = {"source": "fallback", "notes": ["pre_resolution_default"]}
        try:
            fallback_pipeline = default_pipeline_for_route("search_bm25.document_chunk")
            pipeline, pipeline_resolution = _resolve_route_pipeline(
                database,
                route="search_bm25.document_chunk",
                user_name=user_auth.username,
                auth=auth,
                pipeline_override=_pipeline_override,
                fallback_pipeline=fallback_pipeline,
            )
            resolved_pipeline_id = pipeline.pipeline_id
            resolved_pipeline_version = pipeline.version

            def _direct_bm25(**kwargs):
                direct_kwargs = dict(kwargs)
                direct_kwargs["_direct_stage_call"] = True
                direct_kwargs["_skip_observability"] = True
                return search_bm25(**direct_kwargs)

            request_obj = RetrievalRequest(
                route=f"search_bm25.{table}",
                query_text=query,
                collection_ids=collection_ids,
                options={
                    "limit": int(limit),
                    "offset": int(offset),
                    "bm25_query_text": query,
                    "table": table,
                    "sort_by": sort_by,
                    "sort_dir": sort_dir,
                    "web_search": bool(web_search),
                    "_skip_observability": True,
                },
            )
            run_result = _run_async_sync(
                PipelineOrchestrator().run(
                    request=request_obj,
                    pipeline=pipeline,
                    retrievers=_build_retrievers_for_pipeline(
                        pipeline=pipeline,
                        database=database,
                        auth=auth,
                        toolchain_function_caller=None,
                        search_bm25_fn=_direct_bm25,
                        search_hybrid_fn=None,
                    ),
                    fusion=_select_fusion_for_pipeline(pipeline=pipeline, options=request_obj.options),
                    reranker=None,
                    packer=_select_packer(
                        group_chunks=group_chunks,
                        options={**request_obj.options, **({"packing_mode": pipeline.flags.get("packing_mode")} if isinstance(pipeline.flags, dict) and pipeline.flags.get("packing_mode") is not None else {})},
                    ),
                )
            )
            results_made = [_candidate_to_document_chunk(candidate) for candidate in run_result.candidates]
            results_dump = [r.model_dump() for r in results_made]
            if not _skip_observability:
                metrics.record_retrieval(
                    route=f"search_bm25.{table}",
                    status="ok",
                    latency_seconds=time.time() - t_1,
                    results_count=len(results_dump),
                )
                log_retrieval_run(
                    database,
                    route=f"search_bm25.{table}",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=collection_ids,
                    pipeline_id=run_result.pipeline_id,
                    pipeline_version=run_result.pipeline_version,
                    filters={
                        "collection_ids": collection_ids,
                        "web_search": web_search,
                        "table": table,
                        "sort_by": sort_by,
                        "sort_dir": sort_dir,
                        "orchestrated": True,
                    },
                    budgets={"limit": limit, "offset": offset},
                    timings={"total": time.time() - t_1},
                    counters={"rows_returned": len(results_dump), "group_chunks": bool(group_chunks)},
                    result_rows=results_dump,
                    candidate_details=[candidate.model_dump() for candidate in run_result.candidates],
                    status="ok",
                    md={
                        "stage_trace": [trace.model_dump() for trace in run_result.traces],
                        "pipeline_resolution": pipeline_resolution,
                    },
                )
            database.rollback()
            return results_made
        except Exception as e:
            if not _skip_observability:
                metrics.record_retrieval(
                    route=f"search_bm25.{table}",
                    status="error",
                    latency_seconds=time.time() - t_1,
                    results_count=0,
                )
                log_retrieval_run(
                    database,
                    route=f"search_bm25.{table}",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=collection_ids,
                    pipeline_id=resolved_pipeline_id,
                    pipeline_version=resolved_pipeline_version,
                    filters={"collection_ids": collection_ids, "table": table, "orchestrated": True},
                    budgets={"limit": limit, "offset": offset},
                    timings={"total": time.time() - t_1},
                    counters={},
                    result_rows=[],
                    status="error",
                    error=str(e),
                    md={"pipeline_resolution": pipeline_resolution},
                )
            database.rollback()
            raise e

    if web_search:
        collection_ids.append(["WEB"])
    
    collection_string = str(collection_ids).replace("'", "")
    
    formatted_query, strong_where_clause = parse_search(query, valid_fields, catch_all_fields=chosen_catch_alls)
    
    # print("Formatted query:", formatted_query)
    collection_spec = (
        f"""collection_id:IN {str(collection_ids).replace("'", "")}"""
        if (table == "document_chunk")
        else " OR ".join([
            f"""({collection_attr}:IN {str(collection_ids).replace("'", "")})"""
            for collection_attr in document_collection_attrs
        ])
    )
    
    score_field = "paradedb.score(id) AS score, " if formatted_query != "()" else ""
    
    assert not (sort_by == "score" and formatted_query == "()"), \
        "Cannot sort by score if no query is specified"
    
    assert sort_by in valid_fields or sort_by == "score", \
        f"sort_by must be one of {valid_fields}, not {sort_by}"
    
    assert sort_dir in ["DESC", "ASC"], \
        "sort_dir must be either 'DESC' or 'ASC'"
    
    order_by_field = f"ORDER BY {sort_by} {sort_dir}" + (", id ASC" if sort_by != "id" else "")
    
    if table == "segment":
        assert sort_by != "md", "sort_by='md' is not supported for table='segment'"
        segment_collection_filter = ""
        if len(collection_ids) > 0:
            clean_collection_ids = ",".join([f"'{c}'" for c in collection_ids])
            segment_collection_filter = f" AND COALESCE({SegmentTable.__tablename__}.md->>'collection_id', '') IN ({clean_collection_ids})"
        parse_field = formatted_query
    else:
        if formatted_query.startswith("() NOT "):
            formatted_query = formatted_query[3:]
            parse_field = f"({collection_spec}) {formatted_query}"
        else:
            parse_field = f"({collection_spec}) AND ({formatted_query})" \
                if formatted_query != "()" else f"{collection_spec}"
    
    quoted_phrases = _extract_quoted_phrases(query, max_phrases=4)
    phrase_rerank_enabled = (
        formatted_query != "()"
        and sort_by == "score"
        and sort_dir == "DESC"
        and table in {"document_chunk", "segment"}
        and len(quoted_phrases) > 0
    )

    if phrase_rerank_enabled:
        score_bind_params: Dict[str, Any] = {}
        phrase_boost_terms: List[str] = []
        for i, phrase in enumerate(quoted_phrases):
            key = f"phrase_boost_{i}"
            score_bind_params[key] = phrase
            phrase_boost_terms.append(
                f"CASE WHEN POSITION(LOWER(:{key}) IN LOWER({chosen_table.__tablename__}.text)) > 0 THEN 1000 ELSE 0 END"
            )
        phrase_boost_expr = " + ".join(phrase_boost_terms)
        candidate_limit = min(400, max(int(limit) * 8, int(offset) + int(limit) * 4, 80))
        segment_filter_clause = segment_collection_filter if table == "segment" else ""
        STMT = text(f"""
        WITH bm25_candidates AS (
            SELECT id, paradedb.score(id) AS base_score
            FROM {chosen_table.__tablename__}
            WHERE id @@@ paradedb.parse('{parse_field}')
            {segment_filter_clause}
            ORDER BY paradedb.score(id) DESC
            LIMIT :candidate_limit
        )
        SELECT
            {chosen_table.__tablename__}.id AS id,
            (bm25_candidates.base_score + {phrase_boost_expr}) AS score,
            {chosen_attributes}
        FROM bm25_candidates
        JOIN {chosen_table.__tablename__} ON {chosen_table.__tablename__}.id = bm25_candidates.id
        ORDER BY score DESC, {chosen_table.__tablename__}.id ASC
        LIMIT :limit
        OFFSET :offset;
        """).bindparams(
            limit=limit,
            offset=offset,
            candidate_limit=candidate_limit,
            **score_bind_params,
        )
    elif table == "segment":
        STMT = text(f"""
        SELECT id, {score_field}{chosen_attributes}
        FROM {chosen_table.__tablename__}
        WHERE id @@@ paradedb.parse('{parse_field}')
        {segment_collection_filter}
        {order_by_field}
        LIMIT :limit
        OFFSET :offset;
        """).bindparams(
            limit=limit,
            offset=offset,
        )
    else:
        STMT = text(f"""
        SELECT id, {score_field}{chosen_attributes}
        FROM {chosen_table.__tablename__}
        WHERE id @@@ paradedb.parse('{parse_field}')
        {order_by_field}
        LIMIT :limit
        OFFSET :offset;
        """).bindparams(
            limit=limit,
            offset=offset,
        )
    
    
    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))
    
    try:
        subset_start = 1 if formatted_query == "()" else 2
        results = database.exec(STMT)
        results = list(results)
        if table == "document_chunk":
            results_made : List[DocumentChunkDictionary] = list(map(lambda x: convert_chunk_query_result(x[subset_start:]), results))
        elif table == "segment":
            results_made : List[DocumentChunkDictionary] = list(map(lambda x: convert_segment_query_result(x[subset_start:]), results))
        else:
            results_made : List[DocumentRawDictionary] = list(map(lambda x: convert_doc_query_result(x[subset_start:]), results))
        
        if formatted_query != "()":
            for i, chunk in enumerate(results):
                results_made[i].bm25_score = float(chunk[1])
        
        if group_chunks:
            results_made = group_adjacent_chunks(results_made)
        results_made = sorted(results_made, key=lambda x: 0 if x.bm25_score is None else x.bm25_score, reverse=True)
        results_dump = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in results_made]
        if not _skip_observability:
            metrics.record_retrieval(
                route=f"search_bm25.{table}",
                status="ok",
                latency_seconds=time.time() - t_1,
                results_count=len(results_dump),
            )
            log_retrieval_run(
                database,
                route=f"search_bm25.{table}",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=collection_ids,
                pipeline_id=f"orchestrated.search_bm25.{table}.sql",
                pipeline_version="v1",
                filters={
                    "collection_ids": collection_ids,
                    "web_search": web_search,
                    "table": table,
                    "sort_by": sort_by,
                    "sort_dir": sort_dir,
                },
                budgets={"limit": limit, "offset": offset},
                timings={"total": time.time() - t_1},
                counters={
                    "rows_returned": len(results_dump),
                    "group_chunks": bool(group_chunks),
                },
                result_rows=results_dump,
                status="ok",
            )
        database.rollback()
        return results_made
    except Exception as e:
        if not _skip_observability:
            metrics.record_retrieval(
                route=f"search_bm25.{table}",
                status="error",
                latency_seconds=time.time() - t_1,
                results_count=0,
            )
            log_retrieval_run(
                database,
                route=f"search_bm25.{table}",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=collection_ids,
                pipeline_id=f"orchestrated.search_bm25.{table}.sql",
                pipeline_version="v1",
                filters={
                    "collection_ids": collection_ids,
                    "web_search": web_search,
                    "table": table,
                },
                budgets={"limit": limit, "offset": offset},
                timings={"total": time.time() - t_1},
                counters={},
                result_rows=[],
                status="error",
                error=str(e),
            )
        database.rollback()
        raise e
    
def get_random_chunks(database: Session,
                      auth : AuthType,
                      collection_ids: List[str],
                      limit : int = 10) -> List[DocumentChunkDictionary]:
    
    (_, _) = get_user(database, auth)
    
    # assert (isinstance(offset, int) and offset >= 0), \
    #     "offset must be an int greater than 0"
    
    assert (isinstance(limit, int) and limit >= 0 and limit <= 2000), \
        "limit must be an int between 0 and 2000"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    results = database.exec(
        select(*column_attributes)
        .where(
            DocumentChunk.collection_id.in_(collection_ids)
        )
        .order_by(func.random())
        # .offset(offset)
        .limit(limit)
    ).all()
    results = list(results)
    results : List[dict] = list(map(lambda x: convert_chunk_query_result(x, return_wrapped=True), results))
    
    return results
    
def count_chunks(database: Session,
                 auth : AuthType,
                 collection_ids: List[str]) -> int:
    
    (_, _) = get_user(database, auth)
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    count = database.exec(
        select(func.count())
        .where(DocumentChunk.collection_id.in_(collection_ids))
    ).first()
    
    return count


def list_retrieval_runs(
    database: Session,
    auth: AuthType,
    *,
    limit: int = 50,
    route: Optional[str] = None,
    status: Optional[str] = None,
):
    (_, _) = get_user(database, auth)
    assert isinstance(limit, int) and 1 <= limit <= 500, "limit must be an int in [1, 500]"

    stmt = select(RetrievalRunTable).order_by(RetrievalRunTable.created_at.desc()).limit(limit)
    if route is not None:
        stmt = stmt.where(RetrievalRunTable.route == route)
    if status is not None:
        stmt = stmt.where(RetrievalRunTable.status == status)
    rows = list(database.exec(stmt).all())
    return [row.model_dump() for row in rows]


async def replay_retrieval_run(
    database: Session,
    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
    auth: AuthType,
    *,
    run_id: str,
    persist_replay_log: bool = False,
):
    (_, _) = get_user(database, auth)
    assert isinstance(run_id, str) and len(run_id) > 0, "run_id must be a non-empty string"

    run = database.get(RetrievalRunTable, run_id)
    assert run is not None, f"Retrieval run not found: {run_id}"

    route = run.route or ""
    filters = run.filters or {}
    budgets = run.budgets or {}
    counters = run.counters or {}
    query_payload = _decode_query_payload(run.query_text or "")
    pipeline_override = (
        {"pipeline_id": str(run.pipeline_id), "pipeline_version": str(run.pipeline_version)}
        if isinstance(run.pipeline_id, str) and len(run.pipeline_id) > 0
        and isinstance(run.pipeline_version, str) and len(run.pipeline_version) > 0
        else None
    )

    if route == "search_hybrid":
        rerank_enabled = bool(budgets.get("rerank_enabled", False))
        return await search_hybrid(
            database=database,
            toolchain_function_caller=toolchain_function_caller,
            auth=auth,
            query=query_payload,
            collection_ids=filters.get("collection_ids", []),
            limit_bm25=int(budgets.get("limit_bm25", 10)),
            limit_similarity=int(budgets.get("limit_similarity", 10)),
            limit_sparse=int(budgets.get("limit_sparse", 0)),
            similarity_weight=float(counters.get("similarity_weight", 0.1)),
            bm25_weight=float(counters.get("bm25_weight", 0.9)),
            sparse_weight=float(counters.get("sparse_weight", 0.0)),
            use_bm25=bool(counters.get("use_bm25", True)),
            use_similarity=bool(counters.get("use_similarity", True)),
            use_sparse=bool(counters.get("use_sparse", False)),
            sparse_dimensions=int(counters.get("sparse_dimensions", 1024)),
            web_search=bool(filters.get("web_search", False)),
            rerank=rerank_enabled,
            group_chunks=bool(counters.get("group_chunks", True)),
            _skip_observability=(not persist_replay_log),
            _pipeline_override=pipeline_override,
        )

    if route.startswith("search_bm25"):
        if isinstance(query_payload, dict):
            query_text = str(query_payload.get("bm25", query_payload.get("embedding", "")))
        else:
            query_text = str(query_payload)
        return search_bm25(
            database=database,
            auth=auth,
            query=query_text,
            collection_ids=filters.get("collection_ids", []),
            limit=int(budgets.get("limit", 10)),
            offset=int(budgets.get("offset", 0)),
            web_search=bool(filters.get("web_search", False)),
            group_chunks=bool(counters.get("group_chunks", True)),
            table=filters.get("table", "document_chunk"),
            sort_by=filters.get("sort_by", "score"),
            sort_dir=filters.get("sort_dir", "DESC"),
            _skip_observability=(not persist_replay_log),
            _pipeline_override=pipeline_override,
        )

    if route == "search_file_chunks":
        if isinstance(query_payload, dict):
            query_text = str(query_payload.get("bm25", query_payload.get("query", "")))
        else:
            query_text = str(query_payload)
        return search_file_chunks(
            database=database,
            auth=auth,
            query=query_text,
            limit=int(budgets.get("limit", 20)),
            offset=int(budgets.get("offset", 0)),
            sort_by=filters.get("sort_by", "score"),
            sort_dir=filters.get("sort_dir", "DESC"),
            _skip_observability=(not persist_replay_log),
            _pipeline_override=pipeline_override,
        )

    raise ValueError(f"Replay is not implemented for route '{route}'")

# --------------------
# File chunk search
# --------------------

def search_file_chunks(
    database: Session,
    auth: AuthType,
    query: str,
    *,
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "score",
    sort_dir: str = "DESC",
    return_statement: bool = False,
    _direct_stage_call: bool = False,
    _skip_observability: bool = False,
    _pipeline_override: Optional[Dict[str, str]] = None,
):
    t_1 = time.time()
    # Restrict to current user's files
    (_, user_auth) = get_user(database, auth)
    username = user_auth.username

    assert isinstance(query, str), "query must be a string"
    assert (isinstance(limit, int) and 0 <= limit <= 2000), "limit must be 0..2000"
    assert (isinstance(offset, int) and offset >= 0), "offset must be >= 0"

    if not _direct_stage_call and not return_statement:
        resolved_pipeline_id = "orchestrated.search_file_chunks"
        resolved_pipeline_version = "v1"
        pipeline_resolution = {"source": "fallback", "notes": ["pre_resolution_default"]}
        try:
            fallback_pipeline = default_pipeline_for_route("search_file_chunks")
            pipeline, pipeline_resolution = _resolve_route_pipeline(
                database,
                route="search_file_chunks",
                user_name=user_auth.username,
                auth=auth,
                pipeline_override=_pipeline_override,
                fallback_pipeline=fallback_pipeline,
            )
            resolved_pipeline_id = pipeline.pipeline_id
            resolved_pipeline_version = pipeline.version

            request_obj = RetrievalRequest(
                route="search_file_chunks",
                query_text=query,
                collection_ids=[],
                options={
                    "limit": int(limit),
                    "offset": int(offset),
                    "sort_by": sort_by,
                    "sort_dir": sort_dir,
                    "bm25_query_text": query,
                    "_skip_observability": True,
                },
            )

            def _direct_file_chunks(**kwargs):
                return search_file_chunks(
                    **kwargs,
                    _direct_stage_call=True,
                    _skip_observability=True,
                )

            run_result = _run_async_sync(
                PipelineOrchestrator().run(
                    request=request_obj,
                    pipeline=pipeline,
                    retrievers=_build_retrievers_for_pipeline(
                        pipeline=pipeline,
                        database=database,
                        auth=auth,
                        toolchain_function_caller=None,
                        search_bm25_fn=search_bm25,
                        search_hybrid_fn=None,
                        search_file_chunks_fn=_direct_file_chunks,
                    ),
                    fusion=_select_fusion_for_pipeline(pipeline=pipeline, options=request_obj.options),
                    reranker=None,
                    packer=None,
                )
            )
            results = [_candidate_to_file_chunk_result(candidate) for candidate in run_result.candidates]
            if not _skip_observability:
                metrics.record_retrieval(
                    route="search_file_chunks",
                    status="ok",
                    latency_seconds=time.time() - t_1,
                    results_count=len(results),
                )
                log_retrieval_run(
                    database,
                    route="search_file_chunks",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=[],
                    pipeline_id=run_result.pipeline_id,
                    pipeline_version=run_result.pipeline_version,
                    filters={
                        "sort_by": sort_by,
                        "sort_dir": sort_dir,
                        "username_scope": user_auth.username,
                        "orchestrated": True,
                    },
                    budgets={"limit": limit, "offset": offset},
                    timings={"total": time.time() - t_1},
                    counters={"rows_returned": len(results)},
                    result_rows=results,
                    candidate_details=[candidate.model_dump() for candidate in run_result.candidates],
                    status="ok",
                    md={
                        "stage_trace": [trace.model_dump() for trace in run_result.traces],
                        "pipeline_resolution": pipeline_resolution,
                    },
                )
            database.rollback()
            return {"results": results}
        except Exception as e:
            if not _skip_observability:
                metrics.record_retrieval(
                    route="search_file_chunks",
                    status="error",
                    latency_seconds=time.time() - t_1,
                    results_count=0,
                )
                log_retrieval_run(
                    database,
                    route="search_file_chunks",
                    actor_user=user_auth.username,
                    query_payload=query,
                    collection_ids=[],
                    pipeline_id=resolved_pipeline_id,
                    pipeline_version=resolved_pipeline_version,
                    filters={"username_scope": user_auth.username, "orchestrated": True},
                    budgets={"limit": limit, "offset": offset},
                    timings={"total": time.time() - t_1},
                    counters={},
                    result_rows=[],
                    status="error",
                    error=str(e),
                    md={"pipeline_resolution": pipeline_resolution},
                )
            database.rollback()
            raise e

    formatted_query, _ = parse_search(query, FILE_FIELDS, catch_all_fields=["text"])

    score_field = "paradedb.score(fc.id) AS score, " if formatted_query != "()" else ""
    assert not (sort_by == "score" and formatted_query == "()"), "Cannot sort by score if no query is specified"
    assert sort_by in FILE_FIELDS or sort_by == "score", f"sort_by must be one of {FILE_FIELDS} or 'score'"
    assert sort_dir in ["DESC", "ASC"], "sort_dir must be 'DESC' or 'ASC'"

    if sort_by == "score":
        order_by_field = "ORDER BY score DESC, fc.id ASC"
    else:
        order_by_field = f"ORDER BY {sort_by} {sort_dir}" + (", fc.id ASC" if sort_by != "id" else "")

    parse_field = formatted_query if formatted_query != "()" else "()"

    STMT = text(f"""
    SELECT fc.id, {score_field}fc.id, fc.text, fc.md, fc.created_at, fc.file_version_id
    FROM {FileChunkTable.__tablename__} fc
    JOIN {FileVersionTable.__tablename__} fv ON fc.file_version_id = fv.id
    JOIN {FileTable.__tablename__} f ON fv.file_id = f.id
    WHERE f.created_by = :username
      AND fc.id @@@ paradedb.parse('{parse_field}')
    {order_by_field}
    LIMIT :limit
    OFFSET :offset;
    """).bindparams(
        username=username,
        limit=limit,
        offset=offset,
    )

    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))

    try:
        subset_start = 1 if formatted_query == "()" else 2
        rows = list(database.exec(STMT))
        results = []
        for row in rows:
            # row layout: [fc.id, maybe score, fc.id, fc.text, fc.md, fc.created_at, fc.file_version_id]
            idx = subset_start
            results.append(
                {
                    "id": row[idx + 0],
                    "text": row[idx + 1],
                    "md": row[idx + 2],
                    "created_at": float(row[idx + 3]),
                    "file_version_id": row[idx + 4],
                    **({"bm25_score": float(row[1])} if formatted_query != "()" else {}),
                }
            )
        if not _skip_observability:
            metrics.record_retrieval(
                route="search_file_chunks",
                status="ok",
                latency_seconds=time.time() - t_1,
                results_count=len(results),
            )
            log_retrieval_run(
                database,
                route="search_file_chunks",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=[],
                pipeline_id="orchestrated.search_file_chunks.sql",
                pipeline_version="v1",
                filters={
                    "sort_by": sort_by,
                    "sort_dir": sort_dir,
                    "username_scope": user_auth.username,
                },
                budgets={"limit": limit, "offset": offset},
                timings={"total": time.time() - t_1},
                counters={"rows_returned": len(results)},
                result_rows=results,
                status="ok",
            )
        database.rollback()
        return {"results": results}
    except Exception as e:
        if not _skip_observability:
            metrics.record_retrieval(
                route="search_file_chunks",
                status="error",
                latency_seconds=time.time() - t_1,
                results_count=0,
            )
            log_retrieval_run(
                database,
                route="search_file_chunks",
                actor_user=user_auth.username,
                query_payload=query,
                collection_ids=[],
                pipeline_id="orchestrated.search_file_chunks.sql",
                pipeline_version="v1",
                filters={"username_scope": user_auth.username},
                budgets={"limit": limit, "offset": offset},
                timings={"total": time.time() - t_1},
                counters={},
                result_rows=[],
                status="error",
                error=str(e),
            )
        database.rollback()
        raise e

def expand_document_segment(database: Session,
                            auth : AuthType,
                            document_id: str,
                            chunks_to_get: List[int] = [],
                            group_chunks: bool = True,
                            return_embeddings: bool = False) -> DocumentChunkDictionary:
    """
    Get the document chunks for a specific document, in the order specified by the chunks_to_get list.
    i.e. chunks_to_get = [0, 1, 2, 3] will return the first four chunks of the document.
    
    *   group chunks: If True, will group adjacent chunks together into single results, with their chunk numbers as a range, i.e. [0, 3].
    *   return_embeddings: If True, will return the embeddings as a list of floats. 
            doesn't stack perfectly with group chunks, but will still work.
    """
    
    (_, _) = get_user(database, auth)
    
    chunks = list(database.exec(
        # select(*(column_attributes))
        # select(*(column_attributes + [getattr(DocumentChunk, "embedding")]))
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .where(DocumentChunk.document_chunk_number.in_(chunks_to_get))
    ).all())
    
    chunk_tuples = list(map(lambda c: tuple([getattr(c, e) for e in field_strings_no_rerank]), chunks))
    chunks_return = list(map(lambda x: convert_chunk_query_result(x), chunk_tuples))
    # chunks_return = list(map(lambda x: convert_query_result(x[:-1]), chunks))
    
    if return_embeddings:
        for i, chunk in enumerate(chunks):
            chunks_return[i].embedding = chunk.embedding.tolist()
    
    if group_chunks:
        chunks_return = group_adjacent_chunks(chunks_return)
    
    return chunks_return
