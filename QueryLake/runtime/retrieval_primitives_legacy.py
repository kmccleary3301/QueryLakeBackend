from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from sqlmodel import Session

from QueryLake.typing.config import AuthType
from QueryLake.typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalRequest,
)


def _row_to_candidate(row: dict, rank: int, score_key: str, source: str) -> RetrievalCandidate:
    score = row.get(score_key)
    stage_scores = ({score_key: float(score)} if score is not None else {})
    stage_ranks = ({source: rank} if rank > 0 else {})
    return RetrievalCandidate(
        content_id=row["id"] if isinstance(row["id"], str) else row["id"][0],
        text=row.get("text"),
        metadata={
            "creation_timestamp": row.get("creation_timestamp"),
            "collection_type": row.get("collection_type"),
            "document_id": row.get("document_id"),
            "document_name": row.get("document_name"),
            "document_chunk_number": row.get("document_chunk_number"),
            "collection_id": row.get("collection_id"),
            "md": row.get("md", {}),
            "document_md": row.get("document_md", {}),
        },
        stage_scores=stage_scores,
        stage_ranks=stage_ranks,
        provenance=[source],
        selected=True,
    )


class BM25RetrieverParadeDB:
    primitive_id = "BM25RetrieverParadeDB"
    version = "v1"

    def __init__(
        self,
        database: Session,
        auth: AuthType,
        search_bm25_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.database = database
        self.auth = auth
        self.search_bm25_fn = search_bm25_fn

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        limit_key = request.options.get("limit_key")
        offset_key = request.options.get("offset_key")
        limit_raw = request.options.get(limit_key) if isinstance(limit_key, str) and len(limit_key) > 0 else None
        offset_raw = request.options.get(offset_key) if isinstance(offset_key, str) and len(offset_key) > 0 else None
        limit = int(request.options.get("limit", 20) if limit_raw is None else limit_raw)
        offset = int(request.options.get("offset", 0) if offset_raw is None else offset_raw)
        if limit <= 0:
            return []
        if limit <= 0:
            return []
        table = str(request.options.get("table", "document_chunk"))
        sort_by = str(request.options.get("sort_by", "score"))
        sort_dir = str(request.options.get("sort_dir", "DESC")).upper()
        web_search = bool(request.options.get("web_search", False))
        query_key = request.options.get("bm25_query_text_key")
        bm25_query_text = str(
            request.options.get(query_key)
            if isinstance(query_key, str) and len(query_key) > 0 and request.options.get(query_key) is not None
            else request.options.get("bm25_query_text", request.query_text)
        )
        search_bm25_callable = self.search_bm25_fn
        if search_bm25_callable is None:
            from QueryLake.api.search import search_bm25 as search_bm25_callable
        rows = search_bm25_callable(
            database=self.database,
            auth=self.auth,
            query=bm25_query_text,
            collection_ids=request.collection_ids,
            limit=limit,
            offset=offset,
            web_search=web_search,
            table=table,
            sort_by=sort_by,
            sort_dir=sort_dir,
            group_chunks=bool(request.options.get("group_chunks", True)),
            # Primitive wrappers call direct stage execution to avoid recursive orchestration.
            _direct_stage_call=True,
            _skip_observability=bool(request.options.get("_skip_observability", False)),
        )
        row_dicts = [row.model_dump() if hasattr(row, "model_dump") else row for row in rows]
        return [_row_to_candidate(row, i + 1, "bm25_score", "bm25") for i, row in enumerate(row_dicts)]


class DenseRetrieverPGVector:
    primitive_id = "DenseRetrieverPGVector"
    version = "v1"

    def __init__(
        self,
        database: Session,
        auth: AuthType,
        toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
        search_hybrid_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
    ) -> None:
        self.database = database
        self.auth = auth
        self.toolchain_function_caller = toolchain_function_caller
        self.search_hybrid_fn = search_hybrid_fn

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        limit_key = request.options.get("limit_key")
        limit_raw = request.options.get(limit_key) if isinstance(limit_key, str) and len(limit_key) > 0 else None
        limit = int(request.options.get("limit", 20) if limit_raw is None else limit_raw)
        if limit <= 0:
            return []
        query_key = request.options.get("dense_query_text_key")
        dense_query_text = str(
            request.options.get(query_key)
            if isinstance(query_key, str) and len(query_key) > 0 and request.options.get(query_key) is not None
            else request.options.get("dense_query_text", request.query_text)
        )
        search_hybrid_callable = self.search_hybrid_fn
        if search_hybrid_callable is None:
            from QueryLake.api.search import search_hybrid as search_hybrid_callable
        rows = await search_hybrid_callable(
            database=self.database,
            toolchain_function_caller=self.toolchain_function_caller,
            auth=self.auth,
            query={"bm25": dense_query_text, "embedding": dense_query_text},
            embedding=request.query_embedding,
            collection_ids=request.collection_ids,
            limit_bm25=0,
            limit_similarity=limit,
            bm25_weight=0.0,
            similarity_weight=1.0,
            group_chunks=bool(request.options.get("group_chunks", False)),
            # Primitive wrappers call direct stage execution to avoid recursive orchestration.
            _direct_stage_call=True,
            _skip_observability=bool(request.options.get("_skip_observability", False)),
        )
        row_dicts = rows.get("rows", [])
        return [_row_to_candidate(row, i + 1, "similarity_score", "dense") for i, row in enumerate(row_dicts)]


class SparseRetrieverPGVector:
    primitive_id = "SparseRetrieverPGVector"
    version = "v1"

    def __init__(
        self,
        database: Session,
        auth: AuthType,
        toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
        search_hybrid_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
    ) -> None:
        self.database = database
        self.auth = auth
        self.toolchain_function_caller = toolchain_function_caller
        self.search_hybrid_fn = search_hybrid_fn

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        limit_key = request.options.get("limit_key")
        limit_raw = request.options.get(limit_key) if isinstance(limit_key, str) and len(limit_key) > 0 else None
        limit = int(request.options.get("limit", 20) if limit_raw is None else limit_raw)
        if limit <= 0:
            return []
        query_key = request.options.get("sparse_query_text_key")
        sparse_query_text = str(
            request.options.get(query_key)
            if isinstance(query_key, str) and len(query_key) > 0 and request.options.get(query_key) is not None
            else request.options.get("sparse_query_text", request.query_text)
        )
        sparse_query_value_key = request.options.get("sparse_query_value_key")
        sparse_query_value = (
            request.options.get(sparse_query_value_key)
            if isinstance(sparse_query_value_key, str) and len(sparse_query_value_key) > 0
            else request.options.get("sparse_query_value")
        )
        sparse_embedding_function = str(request.options.get("sparse_embedding_function", "embedding_sparse"))
        sparse_dimensions = int(request.options.get("sparse_dimensions", 1024))

        search_hybrid_callable = self.search_hybrid_fn
        if search_hybrid_callable is None:
            from QueryLake.api.search import search_hybrid as search_hybrid_callable
        rows = await search_hybrid_callable(
            database=self.database,
            toolchain_function_caller=self.toolchain_function_caller,
            auth=self.auth,
            query={"bm25": sparse_query_text, "embedding": sparse_query_text, "sparse": sparse_query_text},
            embedding=request.query_embedding,
            embedding_sparse=sparse_query_value,
            collection_ids=request.collection_ids,
            limit_bm25=0,
            limit_similarity=0,
            limit_sparse=limit,
            bm25_weight=0.0,
            similarity_weight=0.0,
            sparse_weight=1.0,
            use_bm25=False,
            use_similarity=False,
            use_sparse=True,
            sparse_embedding_function=sparse_embedding_function,
            sparse_dimensions=sparse_dimensions,
            sparse_prune_max_terms=request.options.get("sparse_prune_max_terms"),
            sparse_prune_min_abs_weight=float(request.options.get("sparse_prune_min_abs_weight", 0.0)),
            sparse_calibration=str(request.options.get("sparse_calibration", "none")),
            group_chunks=bool(request.options.get("group_chunks", False)),
            # Primitive wrappers call direct stage execution to avoid recursive orchestration.
            _direct_stage_call=True,
            _skip_observability=bool(request.options.get("_skip_observability", False)),
        )
        row_dicts = rows.get("rows", [])
        return [_row_to_candidate(row, i + 1, "sparse_score", "sparse") for i, row in enumerate(row_dicts)]


class FileChunkBM25RetrieverSQL:
    primitive_id = "FileChunkBM25RetrieverSQL"
    version = "v1"

    def __init__(
        self,
        database: Session,
        auth: AuthType,
        search_file_chunks_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.database = database
        self.auth = auth
        self.search_file_chunks_fn = search_file_chunks_fn

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        limit_key = request.options.get("limit_key")
        offset_key = request.options.get("offset_key")
        limit_raw = request.options.get(limit_key) if isinstance(limit_key, str) and len(limit_key) > 0 else None
        offset_raw = request.options.get(offset_key) if isinstance(offset_key, str) and len(offset_key) > 0 else None
        limit = int(request.options.get("limit", 20) if limit_raw is None else limit_raw)
        offset = int(request.options.get("offset", 0) if offset_raw is None else offset_raw)
        sort_by = str(request.options.get("sort_by", "score"))
        sort_dir = str(request.options.get("sort_dir", "DESC")).upper()
        query_key = request.options.get("bm25_query_text_key")
        query_text = str(
            request.options.get(query_key)
            if isinstance(query_key, str) and len(query_key) > 0 and request.options.get(query_key) is not None
            else request.options.get("bm25_query_text", request.query_text)
        )
        search_callable = self.search_file_chunks_fn
        if search_callable is None:
            from QueryLake.api.search import search_file_chunks as search_callable
        payload = search_callable(
            database=self.database,
            auth=self.auth,
            query=query_text,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            _direct_stage_call=True,
            _skip_observability=bool(request.options.get("_skip_observability", False)),
        )
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        candidates: List[RetrievalCandidate] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            score = row.get("bm25_score")
            stage_scores = {"bm25_score": float(score)} if isinstance(score, (int, float)) else {}
            candidates.append(
                RetrievalCandidate(
                    content_id=str(row.get("id")),
                    text=row.get("text"),
                    metadata={
                        "md": row.get("md", {}),
                        "created_at": row.get("created_at"),
                        "file_version_id": row.get("file_version_id"),
                    },
                    stage_scores=stage_scores,
                    stage_ranks={"bm25": i + 1},
                    provenance=["bm25"],
                    selected=True,
                )
            )
        return candidates


class RRFusion:
    primitive_id = "RRFusion"
    version = "v1"

    def __init__(self, k: int = 60) -> None:
        self.k = max(1, int(k))

    def fuse(
        self,
        request: RetrievalRequest,
        candidates_by_source: Dict[str, List[RetrievalCandidate]],
    ) -> List[RetrievalCandidate]:
        weights = request.options.get("fusion_weights", {})
        fused: Dict[str, RetrievalCandidate] = {}
        fused_scores: Dict[str, float] = {}

        for source, candidates in candidates_by_source.items():
            source_weight = float(weights.get(source, 1.0))
            for rank, candidate in enumerate(candidates, start=1):
                gain = source_weight * (1.0 / (self.k + rank))
                existing = fused.get(candidate.content_id)
                if existing is None:
                    fused[candidate.content_id] = candidate.model_copy(deep=True)
                    fused_scores[candidate.content_id] = gain
                else:
                    merged = existing.provenance + [p for p in candidate.provenance if p not in existing.provenance]
                    existing.provenance = merged
                    existing.stage_scores = {**existing.stage_scores, **candidate.stage_scores}
                    existing.stage_ranks = {**existing.stage_ranks, **candidate.stage_ranks}
                    fused_scores[candidate.content_id] += gain

        ordered = sorted(
            fused.items(),
            key=lambda item: fused_scores[item[0]],
            reverse=True,
        )
        output: List[RetrievalCandidate] = []
        for rank, (_, candidate) in enumerate(ordered, start=1):
            candidate.stage_scores["rrf_fused"] = fused_scores[candidate.content_id]
            candidate.stage_ranks["rrf_fused"] = rank
            output.append(candidate)
        return output


class WeightedScoreFusion:
    primitive_id = "WeightedScoreFusion"
    version = "v1"

    def __init__(
        self,
        *,
        default_normalization: str = "minmax",
        default_score_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        self.default_normalization = default_normalization
        self.default_score_keys = default_score_keys or {}

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _infer_score_key(candidates: List[RetrievalCandidate], source: str) -> Optional[str]:
        source_prefix = f"{source}_"
        for candidate in candidates:
            for key in candidate.stage_scores.keys():
                if key.startswith(source_prefix):
                    return key
        for candidate in candidates:
            if len(candidate.stage_scores) > 0:
                return next(iter(candidate.stage_scores.keys()))
        return None

    def _normalize_scores(
        self,
        *,
        candidates: List[RetrievalCandidate],
        score_key: Optional[str],
        strategy: str,
    ) -> List[float]:
        fallback_rank = [1.0 / float(i + 1) for i in range(len(candidates))]
        if len(candidates) == 0:
            return []
        if score_key is None:
            return fallback_rank

        raw_scores = [self._safe_float(c.stage_scores.get(score_key)) for c in candidates]
        if any(score is None for score in raw_scores):
            return fallback_rank
        scores = [float(score) for score in raw_scores if score is not None]

        if strategy == "none":
            return scores
        if strategy == "zscore":
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = math.sqrt(variance)
            if std <= 1e-12:
                return [0.5] * len(scores)
            return [1.0 / (1.0 + math.exp(-((s - mean) / std))) for s in scores]
        if strategy == "rank":
            return fallback_rank

        # Default: min-max normalization.
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score <= 1e-12:
            return [1.0] * len(scores)
        scale = max_score - min_score
        return [(s - min_score) / scale for s in scores]

    def fuse(
        self,
        request: RetrievalRequest,
        candidates_by_source: Dict[str, List[RetrievalCandidate]],
    ) -> List[RetrievalCandidate]:
        weights = request.options.get("fusion_weights", {})
        score_keys = {**self.default_score_keys, **request.options.get("fusion_score_keys", {})}
        strategy = str(request.options.get("fusion_normalization", self.default_normalization)).lower().strip()
        if strategy not in {"minmax", "none", "zscore", "rank"}:
            strategy = "minmax"

        fused: Dict[str, RetrievalCandidate] = {}
        fused_scores: Dict[str, float] = {}

        for source, candidates in candidates_by_source.items():
            source_weight = float(weights.get(source, 1.0))
            score_key = score_keys.get(source) or self._infer_score_key(candidates, source)
            normalized_scores = self._normalize_scores(
                candidates=candidates,
                score_key=score_key,
                strategy=strategy,
            )
            for candidate, normalized_score in zip(candidates, normalized_scores):
                contribution = source_weight * normalized_score
                existing = fused.get(candidate.content_id)
                if existing is None:
                    fused[candidate.content_id] = candidate.model_copy(deep=True)
                    fused_scores[candidate.content_id] = contribution
                else:
                    merged = existing.provenance + [p for p in candidate.provenance if p not in existing.provenance]
                    existing.provenance = merged
                    existing.stage_scores = {**existing.stage_scores, **candidate.stage_scores}
                    existing.stage_ranks = {**existing.stage_ranks, **candidate.stage_ranks}
                    fused_scores[candidate.content_id] += contribution

        ordered = sorted(
            fused.items(),
            key=lambda item: (-fused_scores[item[0]], item[0]),
        )
        output: List[RetrievalCandidate] = []
        for rank, (_, candidate) in enumerate(ordered, start=1):
            candidate.stage_scores["weighted_fused"] = fused_scores[candidate.content_id]
            candidate.stage_ranks["weighted_fused"] = rank
            output.append(candidate)
        return output


class AdjacentChunkPacker:
    primitive_id = "AdjacentChunkPacker"
    version = "v1"

    @staticmethod
    def _chunk_bounds(value: Any) -> Optional[tuple[int, int]]:
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                low = int(value[0])
                high = int(value[1])
                if low > high:
                    low, high = high, low
                return (low, high)
            except Exception:
                return None
        return None

    @staticmethod
    def _overlap_chars(left: str, right: str) -> int:
        max_overlap = min(len(left), len(right))
        for i in range(max_overlap, 0, -1):
            if left[-i:] == right[:i]:
                return i
        return 0

    def pack(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates

        max_gap = int(request.options.get("adjacent_max_gap", 1))
        min_overlap = int(request.options.get("adjacent_min_overlap_chars", 100))

        grouped: Dict[str, List[tuple[int, RetrievalCandidate, tuple[int, int]]]] = {}
        passthrough: List[tuple[int, RetrievalCandidate]] = []
        for idx, candidate in enumerate(candidates):
            document_id = candidate.metadata.get("document_id")
            bounds = self._chunk_bounds(candidate.metadata.get("document_chunk_number"))
            if not isinstance(document_id, str) or bounds is None:
                passthrough.append((idx, candidate))
                continue
            grouped.setdefault(document_id, []).append((idx, candidate, bounds))

        merged_ranked: List[tuple[int, RetrievalCandidate]] = passthrough[:]
        for rows in grouped.values():
            rows_sorted = sorted(rows, key=lambda row: row[2][0])
            seed_idx, seed_candidate, seed_bounds = rows_sorted[0]
            current = seed_candidate.model_copy(deep=True)
            current_low, current_high = seed_bounds
            member_ids = [seed_candidate.content_id]
            current_rank = seed_idx

            for idx, candidate, bounds in rows_sorted[1:]:
                next_low, next_high = bounds
                if next_low <= current_high + max_gap:
                    overlap = self._overlap_chars(current.text or "", candidate.text or "")
                    if overlap >= min_overlap:
                        current.text = (current.text or "") + (candidate.text or "")[overlap:]
                    else:
                        current.text = ((current.text or "").rstrip() + "\n\n" + (candidate.text or "").lstrip()).strip()
                    current_low = min(current_low, next_low)
                    current_high = max(current_high, next_high)
                    member_ids.append(candidate.content_id)
                    current.provenance = current.provenance + [p for p in candidate.provenance if p not in current.provenance]
                    for score_key, score_value in candidate.stage_scores.items():
                        if score_key not in current.stage_scores:
                            current.stage_scores[score_key] = score_value
                        else:
                            current.stage_scores[score_key] = max(current.stage_scores[score_key], score_value)
                    for rank_key, rank_value in candidate.stage_ranks.items():
                        if rank_key not in current.stage_ranks:
                            current.stage_ranks[rank_key] = rank_value
                        else:
                            current.stage_ranks[rank_key] = min(current.stage_ranks[rank_key], rank_value)
                    current_rank = min(current_rank, idx)
                else:
                    current.metadata["document_chunk_number"] = (current_low, current_high)
                    current.metadata["merged_content_ids"] = member_ids
                    merged_ranked.append((current_rank, current))
                    current = candidate.model_copy(deep=True)
                    current_low, current_high = bounds
                    member_ids = [candidate.content_id]
                    current_rank = idx

            current.metadata["document_chunk_number"] = (current_low, current_high)
            current.metadata["merged_content_ids"] = member_ids
            merged_ranked.append((current_rank, current))

        merged_ranked.sort(key=lambda row: row[0])
        return [candidate for _, candidate in merged_ranked]


class DiversityAwarePacker:
    primitive_id = "DiversityAwarePacker"
    version = "v1"

    def pack(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates
        max_per_document = max(1, int(request.options.get("max_per_document", 1)))
        max_total = max(1, int(request.options.get("limit", len(candidates))))
        selected: List[RetrievalCandidate] = []
        doc_counts: Dict[str, int] = {}

        for candidate in candidates:
            doc_key = (
                candidate.metadata.get("document_id")
                or candidate.metadata.get("document_name")
                or candidate.content_id
            )
            doc_key = str(doc_key)
            if doc_counts.get(doc_key, 0) >= max_per_document:
                continue
            selected.append(candidate)
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            if len(selected) >= max_total:
                break
        return selected


class TokenBudgetPacker:
    primitive_id = "TokenBudgetPacker"
    version = "v1"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Keep estimator deterministic and lightweight for routing-stage packing.
        token_estimate = len(text.split())
        return max(1, int(token_estimate))

    def pack(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates
        token_budget = max(1, int(request.options.get("token_budget", 1200)))
        selected: List[RetrievalCandidate] = []
        consumed = 0
        for candidate in candidates:
            estimate = self._estimate_tokens(candidate.text or "")
            if len(selected) > 0 and consumed + estimate > token_budget:
                break
            consumed += estimate
            next_candidate = candidate.model_copy(deep=True)
            next_candidate.metadata = {
                **(next_candidate.metadata or {}),
                "estimated_tokens": estimate,
                "token_budget_running_total": consumed,
            }
            selected.append(next_candidate)
        return selected


class CitationAwarePacker:
    primitive_id = "CitationAwarePacker"
    version = "v1"

    @staticmethod
    def _citation_features(candidate: RetrievalCandidate) -> tuple[int, float]:
        md = candidate.metadata or {}
        citation_count_raw = md.get("citation_count", 0)
        citation_score_raw = md.get("citation_score", 0.0)
        has_citation_raw = md.get("has_citation", None)
        try:
            citation_count = int(citation_count_raw)
        except Exception:
            citation_count = 0
        try:
            citation_score = float(citation_score_raw)
        except Exception:
            citation_score = 0.0
        has_citation = bool(has_citation_raw) or citation_count > 0 or citation_score > 0.0
        return (1 if has_citation else 0, citation_score)

    @staticmethod
    def _best_rank(candidate: RetrievalCandidate) -> int:
        if len(candidate.stage_ranks) == 0:
            return 10**9
        return min(int(v) for v in candidate.stage_ranks.values())

    def pack(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates
        limit = max(1, int(request.options.get("limit", len(candidates))))
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -self._citation_features(candidate)[0],
                -self._citation_features(candidate)[1],
                self._best_rank(candidate),
                candidate.content_id,
            ),
        )
        return ranked[:limit]


class CrossEncoderReranker:
    primitive_id = "CrossEncoderReranker"
    version = "v1"

    def __init__(self, auth: AuthType, toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]]) -> None:
        self.auth = auth
        self.toolchain_function_caller = toolchain_function_caller

    async def rerank(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if len(candidates) == 0:
            return []
        rerank_query = str(request.options.get("rerank_query_text", request.query_text))
        rerank_call: Awaitable[Callable] = self.toolchain_function_caller("rerank")
        scores = await rerank_call(
            self.auth,
            [(rerank_query, candidate.text or "") for candidate in candidates],
        )
        ranked = []
        for i, candidate in enumerate(candidates):
            next_candidate = candidate.model_copy(deep=True)
            next_candidate.stage_scores["rerank_score"] = float(scores[i])
            ranked.append(next_candidate)
        ranked = sorted(ranked, key=lambda c: c.stage_scores.get("rerank_score", 0.0), reverse=True)
        for rank, candidate in enumerate(ranked, start=1):
            candidate.stage_ranks["rerank"] = rank
        return ranked
