import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import (  # noqa: E402
    _estimate_reserved_vram,
    _recommend_gpu_memory_fraction,
    _select_gpu_fraction,
)


def test_recommend_gpu_memory_fraction_accounts_for_context():
    per_node_capacity = 49_140  # 49 GB GPU
    base_vram = 19_656  # ~19.6 GB weights
    max_ctx = 32_768

    recommended = _recommend_gpu_memory_fraction(
        required_vram=base_vram,
        max_model_len=max_ctx,
        per_node_capacity=per_node_capacity,
        configured_fraction=0.2,  # clearly too low
    )

    # Expect roughly 0.625 (0.4 base + ~0.225 context bonus)
    assert recommended == pytest.approx(0.625, rel=1e-3)


def test_select_gpu_fraction_applies_safety_buffer():
    buffered_fraction = _select_gpu_fraction(existing_fraction=0.4, recommended_fraction=0.62)
    assert buffered_fraction == pytest.approx(0.67, rel=1e-3)


def test_estimate_reserved_vram_matches_larger_fraction():
    per_node_capacity = 49_140
    reserved = _estimate_reserved_vram(
        per_node_capacity=per_node_capacity,
        required_vram=19_656,
        util_fraction=0.62,
        gpu_fraction=0.67,
    )
    expected = math.ceil(per_node_capacity * 0.67)
    assert reserved == expected
