from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    content_id: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stage_scores: Dict[str, float] = Field(default_factory=dict)
    stage_ranks: Dict[str, int] = Field(default_factory=dict)
    provenance: List[str] = Field(default_factory=list)
    selected: bool = True


class RetrievalRequest(BaseModel):
    route: str = "search"
    query_text: str
    query_embedding: Optional[List[float]] = None
    collection_ids: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    budgets: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    actor_user: Optional[str] = None


class RetrievalStageTrace(BaseModel):
    stage: str
    duration_ms: float = 0.0
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class RetrievalPipelineStage(BaseModel):
    stage_id: str
    primitive_id: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RetrievalPipelineSpec(BaseModel):
    pipeline_id: str
    version: str
    stages: List[RetrievalPipelineStage] = Field(default_factory=list)
    budgets: Dict[str, Any] = Field(default_factory=dict)
    flags: Dict[str, Any] = Field(default_factory=dict)


class RetrievalExecutionResult(BaseModel):
    pipeline_id: str
    pipeline_version: str
    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    traces: List[RetrievalStageTrace] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRoutingDecision(BaseModel):
    route_class: str  # keyword_heavy | semantic | multi_hop | custom
    target_pipeline_id: Optional[str] = None
    target_pipeline_version: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    options_overrides: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RetrieverPrimitive(Protocol):
    primitive_id: str
    version: str

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class FusionPrimitive(Protocol):
    primitive_id: str
    version: str

    def fuse(
        self,
        request: RetrievalRequest,
        candidates_by_source: Dict[str, List[RetrievalCandidate]],
    ) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class RerankerPrimitive(Protocol):
    primitive_id: str
    version: str

    async def rerank(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class ContextPackingPrimitive(Protocol):
    primitive_id: str
    version: str

    def pack(
        self,
        request: RetrievalRequest,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class QueryRouterPrimitive(Protocol):
    primitive_id: str
    version: str

    def route(self, request: RetrievalRequest) -> QueryRoutingDecision:
        ...


@runtime_checkable
class LearnedSparseRetrieverPrimitive(Protocol):
    primitive_id: str
    version: str

    async def retrieve_sparse(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class LateInteractionRetrieverPrimitive(Protocol):
    primitive_id: str
    version: str

    async def retrieve_late_interaction(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        ...


@runtime_checkable
class GraphRetrieverPrimitive(Protocol):
    primitive_id: str
    version: str

    async def retrieve_graph(self, request: RetrievalRequest) -> List[RetrievalCandidate]:
        ...
