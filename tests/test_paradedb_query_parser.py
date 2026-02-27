from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.misc_functions.paradedb_query_parser import parse_search
from QueryLake.database.sql_db_tables import CHUNK_INDEXED_COLUMNS


def test_parse_search_strips_sqlish_punctuation():
    query = "text:hello'; DROP TABLE users; --"
    parsed, _ = parse_search(query, CHUNK_INDEXED_COLUMNS, catch_all_fields=["text"])
    assert ";" not in parsed
    assert "'" not in parsed


def test_parse_search_removes_non_printable_characters():
    query = "boiler\x00pressure\x01limits"
    parsed, _ = parse_search(query, CHUNK_INDEXED_COLUMNS, catch_all_fields=["text"])
    assert "\x00" not in parsed
    assert "\x01" not in parsed


def test_parse_search_applies_input_length_guard():
    very_long = "alpha " * 5000
    parsed, _ = parse_search(very_long, CHUNK_INDEXED_COLUMNS, catch_all_fields=["text"])
    # Should parse without exploding query length.
    assert len(parsed) < 120000
