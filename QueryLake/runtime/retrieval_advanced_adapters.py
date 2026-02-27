from __future__ import annotations

import re
from typing import Awaitable, Callable, List, Optional

from QueryLake.typing.retrieval_primitives import (
    QueryRoutingDecision,
    RetrievalCandidate,
    RetrievalRequest,
)


def classify_query_text(query_text: str) -> str:
    query = (query_text or "").strip()
    lowered = query.lower()
    if len(query) == 0:
        return "semantic"

    keyword_markers = [
        "site:",
        "filetype:",
        "\"",
        " and ",
        " or ",
        " not ",
        "+",
        "-",
    ]
    if any(marker in lowered for marker in keyword_markers):
        return "keyword_heavy"
    if re.search(r"[A-Z]{2,}_[A-Z0-9_]+", query) is not None:
        return "keyword_heavy"

    multihop_markers = [
        "compare",
        "versus",
        "vs.",
        "relationship",
        "tradeoff",
        "root cause",
        "why",
        "how do",
        "and then",
    ]
    if len(query.split()) >= 8 and any(marker in lowered for marker in multihop_markers):
        return "multi_hop"
    return "semantic"


class HeuristicQueryRouter:
    primitive_id = "HeuristicQueryRouter"
    version = "v1"

    def __init__(
        self,
        *,
        keyword_pipeline_id: str = "orchestrated.search_bm25.document_chunk",
        keyword_pipeline_version: str = "v1",
        semantic_pipeline_id: str = "orchestrated.search_hybrid",
        semantic_pipeline_version: str = "v1",
        multihop_pipeline_id: str = "orchestrated.search_hybrid",
        multihop_pipeline_version: str = "v1",
    ) -> None:
        self.keyword_pipeline_id = keyword_pipeline_id
        self.keyword_pipeline_version = keyword_pipeline_version
        self.semantic_pipeline_id = semantic_pipeline_id
        self.semantic_pipeline_version = semantic_pipeline_version
        self.multihop_pipeline_id = multihop_pipeline_id
        self.multihop_pipeline_version = multihop_pipeline_version

    def route(self, request: RetrievalRequest) -> QueryRoutingDecision:
        route_class = classify_query_text(request.query_text)
        if route_class == "keyword_heavy":
            return QueryRoutingDecision(
                route_class=route_class,
                target_pipeline_id=self.keyword_pipeline_id,
                target_pipeline_version=self.keyword_pipeline_version,
                reasons=["keyword_markers"],
                options_overrides={"prefer_bm25": True},
            )
        if route_class == "multi_hop":
            return QueryRoutingDecision(
                route_class=route_class,
                target_pipeline_id=self.multihop_pipeline_id,
                target_pipeline_version=self.multihop_pipeline_version,
                reasons=["multihop_markers"],
                options_overrides={"max_hops": 3, "planner_mode": "decompose"},
            )
        return QueryRoutingDecision(
            route_class="semantic",
            target_pipeline_id=self.semantic_pipeline_id,
            target_pipeline_version=self.semantic_pipeline_version,
            reasons=["default_semantic"],
            options_overrides={"prefer_hybrid": True},
        )


class LearnedSparseHookRetriever:
    primitive_id = "LearnedSparseHookRetriever"
    version = "v1"

    def __init__(
        self,
        *,
        provider: Optional[Callable[[RetrievalRequest], Awaitable[List[RetrievalCandidate]]]] = None,
    ) -> None:
        self.provider = provider

    async def retrieve_sparse(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        if self.provider is None:
            return []
        rows = await self.provider(request)
        return list(rows or [])


class LateInteractionRetrieverAdapter:
    primitive_id = "LateInteractionRetrieverAdapter"
    version = "v1"

    def __init__(
        self,
        *,
        provider: Optional[Callable[[RetrievalRequest], Awaitable[List[RetrievalCandidate]]]] = None,
    ) -> None:
        self.provider = provider

    async def retrieve_late_interaction(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        if self.provider is None:
            return []
        rows = await self.provider(request)
        return list(rows or [])


class GraphRetrieverAdapter:
    primitive_id = "GraphRetrieverAdapter"
    version = "v1"

    def __init__(
        self,
        *,
        provider: Optional[Callable[[RetrievalRequest], Awaitable[List[RetrievalCandidate]]]] = None,
    ) -> None:
        self.provider = provider

    async def retrieve_graph(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        if self.provider is None:
            return []
        rows = await self.provider(request)
        return list(rows or [])
