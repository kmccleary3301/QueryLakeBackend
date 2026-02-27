import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_advanced_adapters import (
    GraphRetrieverAdapter,
    HeuristicQueryRouter,
    LateInteractionRetrieverAdapter,
    LearnedSparseHookRetriever,
    classify_query_text,
)
from QueryLake.typing.retrieval_primitives import RetrievalCandidate, RetrievalRequest


def test_classify_query_text_routes_keyword_and_multihop():
    assert classify_query_text('site:docs.basf.com "vapor recovery"') == "keyword_heavy"
    assert classify_query_text("Compare vapor recovery versus mineral oil procedures across units") == "multi_hop"
    assert classify_query_text("vapor recovery best practice") == "semantic"


def test_heuristic_query_router_returns_pipeline_and_overrides():
    router = HeuristicQueryRouter()
    decision = router.route(RetrievalRequest(query_text='site:docs.basf.com "boiler pressure"'))
    assert decision.route_class == "keyword_heavy"
    assert decision.target_pipeline_id == "orchestrated.search_bm25.document_chunk"
    assert decision.options_overrides["prefer_bm25"] is True


def test_advanced_retriever_adapters_call_provider():
    async def _provider(request: RetrievalRequest):
        return [RetrievalCandidate(content_id="c1", text=request.query_text)]

    request = RetrievalRequest(query_text="hello")
    sparse = LearnedSparseHookRetriever(provider=_provider)
    late = LateInteractionRetrieverAdapter(provider=_provider)
    graph = GraphRetrieverAdapter(provider=_provider)

    sparse_rows = asyncio.run(sparse.retrieve_sparse(request))
    late_rows = asyncio.run(late.retrieve_late_interaction(request))
    graph_rows = asyncio.run(graph.retrieve_graph(request))

    assert sparse_rows[0].content_id == "c1"
    assert late_rows[0].text == "hello"
    assert graph_rows[0].content_id == "c1"


def test_advanced_retriever_adapters_fallback_to_empty_without_provider():
    request = RetrievalRequest(query_text="hello")
    sparse = LearnedSparseHookRetriever()
    late = LateInteractionRetrieverAdapter()
    graph = GraphRetrieverAdapter()

    assert asyncio.run(sparse.retrieve_sparse(request)) == []
    assert asyncio.run(late.retrieve_late_interaction(request)) == []
    assert asyncio.run(graph.retrieve_graph(request)) == []
