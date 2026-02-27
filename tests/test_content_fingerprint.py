from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.content_fingerprint import content_fingerprint


def test_content_fingerprint_is_deterministic():
    h1 = content_fingerprint(text="alpha", md={"a": 1, "b": 2})
    h2 = content_fingerprint(text="alpha", md={"b": 2, "a": 1})
    assert h1 == h2


def test_content_fingerprint_changes_on_text_or_md_change():
    base = content_fingerprint(text="alpha", md={"a": 1})
    changed_text = content_fingerprint(text="beta", md={"a": 1})
    changed_md = content_fingerprint(text="alpha", md={"a": 2})
    assert base != changed_text
    assert base != changed_md
