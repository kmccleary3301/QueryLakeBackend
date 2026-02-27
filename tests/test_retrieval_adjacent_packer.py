from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_primitives_legacy import (
    AdjacentChunkPacker,
    CitationAwarePacker,
    DiversityAwarePacker,
    TokenBudgetPacker,
)
from QueryLake.typing.retrieval_primitives import RetrievalCandidate, RetrievalRequest


def test_adjacent_chunk_packer_merges_contiguous_chunks():
    packer = AdjacentChunkPacker()
    request = RetrievalRequest(
        query_text="q",
        options={"adjacent_max_gap": 1, "adjacent_min_overlap_chars": 5},
    )
    candidates = [
        RetrievalCandidate(
            content_id="c1",
            text="alpha beta",
            metadata={"document_id": "d1", "document_chunk_number": 0},
            stage_scores={"bm25_score": 0.7},
            stage_ranks={"bm25": 2},
            provenance=["bm25"],
        ),
        RetrievalCandidate(
            content_id="c2",
            text=" beta gamma",
            metadata={"document_id": "d1", "document_chunk_number": 1},
            stage_scores={"bm25_score": 0.9},
            stage_ranks={"bm25": 1},
            provenance=["bm25"],
        ),
    ]

    packed = packer.pack(request, candidates)
    assert len(packed) == 1
    assert packed[0].metadata["document_chunk_number"] == (0, 1)
    assert packed[0].metadata["merged_content_ids"] == ["c1", "c2"]
    assert "gamma" in (packed[0].text or "")
    assert packed[0].stage_scores["bm25_score"] == 0.9
    assert packed[0].stage_ranks["bm25"] == 1


def test_adjacent_chunk_packer_keeps_nonadjacent_or_cross_doc_rows_separate():
    packer = AdjacentChunkPacker()
    request = RetrievalRequest(query_text="q", options={"adjacent_max_gap": 1})
    candidates = [
        RetrievalCandidate(
            content_id="a",
            text="row a",
            metadata={"document_id": "d1", "document_chunk_number": 0},
            provenance=["bm25"],
        ),
        RetrievalCandidate(
            content_id="b",
            text="row b",
            metadata={"document_id": "d1", "document_chunk_number": 3},
            provenance=["bm25"],
        ),
        RetrievalCandidate(
            content_id="c",
            text="row c",
            metadata={"document_id": "d2", "document_chunk_number": 1},
            provenance=["dense"],
        ),
    ]

    packed = packer.pack(request, candidates)
    assert [row.content_id for row in packed] == ["a", "b", "c"]


def test_diversity_aware_packer_limits_per_document():
    packer = DiversityAwarePacker()
    request = RetrievalRequest(query_text="q", options={"max_per_document": 1, "limit": 3})
    candidates = [
        RetrievalCandidate(content_id="a1", text="a1", metadata={"document_id": "d1"}),
        RetrievalCandidate(content_id="a2", text="a2", metadata={"document_id": "d1"}),
        RetrievalCandidate(content_id="b1", text="b1", metadata={"document_id": "d2"}),
    ]
    packed = packer.pack(request, candidates)
    assert [row.content_id for row in packed] == ["a1", "b1"]


def test_token_budget_packer_respects_budget():
    packer = TokenBudgetPacker()
    request = RetrievalRequest(query_text="q", options={"token_budget": 3})
    candidates = [
        RetrievalCandidate(content_id="a", text="one two", metadata={}),
        RetrievalCandidate(content_id="b", text="three four", metadata={}),
    ]
    packed = packer.pack(request, candidates)
    assert [row.content_id for row in packed] == ["a"]
    assert packed[0].metadata["estimated_tokens"] == 2


def test_citation_aware_packer_prioritizes_cited_rows():
    packer = CitationAwarePacker()
    request = RetrievalRequest(query_text="q", options={"limit": 2})
    candidates = [
        RetrievalCandidate(
            content_id="a",
            text="a",
            metadata={"citation_count": 0, "citation_score": 0.0},
            stage_ranks={"bm25": 1},
        ),
        RetrievalCandidate(
            content_id="b",
            text="b",
            metadata={"citation_count": 2, "citation_score": 0.8},
            stage_ranks={"bm25": 3},
        ),
        RetrievalCandidate(
            content_id="c",
            text="c",
            metadata={"citation_count": 1, "citation_score": 0.3},
            stage_ranks={"bm25": 2},
        ),
    ]
    packed = packer.pack(request, candidates)
    assert [row.content_id for row in packed] == ["b", "c"]
