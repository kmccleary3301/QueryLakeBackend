import re
from typing import List

from ..database.sql_db_tables import CHUNK_INDEXED_COLUMNS

VALID_FIELDS = CHUNK_INDEXED_COLUMNS

_QUOTE_SEGMENT_RE = re.compile(r"\"([^\"]*)\"(\~\d+)?(\^\d+(\.\d+)?)?")
_QUOTE_CLEAN_RE = re.compile(r"(\'|\")")
_FIELD_RE = re.compile(r"^([a-zA-Z0-9_.]+)\:")
_BOOST_RE = re.compile(r"\^(\d+(\.\d+)?)$")
_SLOP_RE = re.compile(r"\~(\d+)$")
_FORBIDDEN_CHARS = "\\+^`:{}[]()~!*',|&<>?/=;$"
_SANITIZE_TABLE = str.maketrans("", "", _FORBIDDEN_CHARS)


def field_match_format(field: str, value: str) -> str:
    """Format a field match."""
    return f"{field}:{value}"


def _append_unique(values: List[str], seen: set, candidate: str) -> None:
    if candidate in seen:
        return
    seen.add(candidate)
    values.append(candidate)


def parse_search(
    text_in: str,
    valid_fields: List[str],
    catch_all_fields: List[str] = ["text"],
    return_id_exclusions: bool = False,
    return_everything: bool = False,
) -> str:
    assert isinstance(text_in, str), "Search text must be a string"
    # Guardrails for parser stability/safety.
    text_in = text_in[:4000]
    text_in = "".join(ch if ch.isprintable() else " " for ch in text_in)
    text_in = text_in.replace("AND", "and").replace("OR", "or").replace("NOT", "not").replace("\n", " ")

    id_exclusions = []

    TWO_SEQUENCE_SLOP = 2
    THREE_SEQUENCE_SLOP = 3

    TWO_SEQUENCE_BOOST = 20
    THREE_SEQUENCE_BOOST = 60
    QUOTED_PHRASE_DEFAULT_BOOST = 40

    assert all([f in valid_fields for f in catch_all_fields]), (
        f"Invalid field(s) {str([f for f in catch_all_fields if not f in valid_fields])} "
        f"in catch_all_fields. Valid fields are: {valid_fields}"
    )

    call = text_in
    phrase_arguments = {}
    quote_segments = _QUOTE_SEGMENT_RE.finditer(call)

    for i, segment in enumerate(list(quote_segments)):
        phrase_arguments[f"quote_arg_{i}"] = segment.group(0)

    for key, value in phrase_arguments.items():
        call = call.replace(key, f"%{key}")
        call = call.replace(value, key)

    call = _QUOTE_CLEAN_RE.sub(" ", call)
    terms = call.split(" ")

    necessary_args = []
    args_parsed, args_parsed_negative = [], []
    term_sequences, current_term_sequence = [], []
    two_term_sequences, three_term_sequences = [], []

    # Process terms of the search.
    # Term sequences are used to boost the score of terms that appear together, similar to cover density.
    for term in terms:
        term_is_quote_arg, slop, boost, field = False, 1, 1, None

        negative = term.startswith("-")
        term = term.strip("-")

        field_specified = _FIELD_RE.search(term)
        if field_specified:
            field = field_specified.group(1)
            term = term[len(field) + 1 :]

        if not (field in valid_fields or (isinstance(field, str) and field.split(".")[0] in valid_fields)):
            field = None

        # Check if the term was moved to a parsed argument, set flag if so.
        if term in phrase_arguments:
            term_is_quote_arg = term
            term = phrase_arguments[term]

        boost_specified = _BOOST_RE.search(term)
        if boost_specified:
            boost = float(boost_specified.group(1))
            term = term[: boost_specified.start()]
        elif term_is_quote_arg and (not negative):
            # Encourage exact-phrase hits to rank above loose term matches
            # without needing a post-filter text scan.
            boost = QUOTED_PHRASE_DEFAULT_BOOST

        slop_specified = _SLOP_RE.search(term)
        if slop_specified:
            slop = int(slop_specified.group(1))
            term = term[: slop_specified.start()]

        # Wipe special characters. Escaping can be restored once token escaping is re-enabled.
        term = term.translate(_SANITIZE_TABLE)

        # If term was a parsed argument, perform preprocessing.
        if term_is_quote_arg:
            term = term.strip("\"")
            term = term.replace("\"", " ")
            term = f"\"{term.strip()}\""

        if term.strip() in ["", "\"\""]:
            continue

        # Add slop to phrase query.
        if slop != 1 and term_is_quote_arg and not negative:
            term = f"{term}~{slop}"

        # Add boost to query.
        if boost != 1 and not negative:
            term = f"{term}^{boost}"

        # If not negative and not a quote sequence, add to current term sequence.
        if (not negative) and (not term_is_quote_arg) and (field is None):
            current_term_sequence.append(term)
        else:
            term_sequences.append(current_term_sequence)
            current_term_sequence = []

        # Route parsed terms.
        if negative:
            args_parsed_negative.append((term, field))
        elif field is not None:
            necessary_args.append((term, field))
        else:
            args_parsed.append((term, field))

    term_sequences.append(current_term_sequence)

    for term_sequence in term_sequences:
        if len(term_sequence) > 1:
            for i in range(1, len(term_sequence)):
                two_term_sequences.append(f"{term_sequence[i - 1]} {term_sequence[i]}")
        if len(term_sequence) > 2:
            for i in range(2, len(term_sequence)):
                three_term_sequences.append(f"{term_sequence[i - 2]} {term_sequence[i - 1]} {term_sequence[i]}")

    for e in two_term_sequences:
        args_parsed.append((f"\"{e}\"~{TWO_SEQUENCE_SLOP}^{TWO_SEQUENCE_BOOST:.2f}", None))

    for e in three_term_sequences:
        args_parsed.append((f"\"{e}\"~{THREE_SEQUENCE_SLOP}^{THREE_SEQUENCE_BOOST:.2f}", None))

    p_fields: List[str] = []
    n_fields: List[str] = []
    p_seen = set()
    n_seen = set()

    for (e, field) in args_parsed:
        if field is None:
            for catch_all_field in catch_all_fields:
                _append_unique(p_fields, p_seen, field_match_format(catch_all_field, e))
        else:
            _append_unique(p_fields, p_seen, field_match_format(field, e))

    for (e, field) in args_parsed_negative:
        if field is None:
            for catch_all_field in catch_all_fields:
                _append_unique(n_fields, n_seen, field_match_format(catch_all_field, e))
        else:
            _append_unique(n_fields, n_seen, field_match_format(field, e))
            if field == "id":
                id_exclusions.append(e.strip("\""))

    negative_field = " NOT ".join(n_fields)
    positive_field = " OR ".join(p_fields)
    single_condition = f"({positive_field}) NOT {negative_field}" if len(args_parsed_negative) > 0 else f"({positive_field})"

    if len(necessary_args) > 0:
        necessary_unique = list(dict.fromkeys([field_match_format(field, term) for term, field in necessary_args]))
        necessary_joined = " AND ".join(necessary_unique)
        final_query = (
            f"{necessary_joined} AND {single_condition}"
            if (len(args_parsed) + len(args_parsed_negative)) > 0
            else necessary_joined
        )
        strong_where_clause = necessary_joined + f" NOT {negative_field}" if args_parsed_negative else necessary_joined
    else:
        final_query = single_condition
        strong_where_clause = f"NOT {negative_field}" if args_parsed_negative else None

    if return_id_exclusions:
        return final_query, strong_where_clause, id_exclusions
    elif return_everything:
        return (
            final_query,
            strong_where_clause,
            necessary_args,
            n_fields,
            p_fields,
            args_parsed,
            args_parsed_negative,
            term_sequences,
        )

    return final_query, strong_where_clause
