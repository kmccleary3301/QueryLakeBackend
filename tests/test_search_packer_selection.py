from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api.search import _select_packer
from QueryLake.runtime.retrieval_primitives_legacy import (
    AdjacentChunkPacker,
    CitationAwarePacker,
    DiversityAwarePacker,
    TokenBudgetPacker,
)


def test_select_packer_defaults_to_adjacent_for_group_chunks(monkeypatch):
    monkeypatch.delenv("QUERYLAKE_RETRIEVAL_PACKER_VARIANT", raising=False)
    packer = _select_packer(group_chunks=True, options={})
    assert isinstance(packer, AdjacentChunkPacker)


def test_select_packer_uses_env_variant(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_PACKER_VARIANT", "diversity")
    packer = _select_packer(group_chunks=True, options={})
    assert isinstance(packer, DiversityAwarePacker)


def test_select_packer_uses_request_option_override(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_PACKER_VARIANT", "adjacent")
    packer = _select_packer(group_chunks=True, options={"packing_mode": "citation"})
    assert isinstance(packer, CitationAwarePacker)

    packer = _select_packer(group_chunks=False, options={"packing_mode": "token_budget"})
    assert isinstance(packer, TokenBudgetPacker)


def test_select_packer_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("QUERYLAKE_RETRIEVAL_PACKER_VARIANT", raising=False)
    assert _select_packer(group_chunks=False, options={"packing_mode": "none"}) is None
