#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class PageMetrics:
    page: str
    baseline_chars: int
    candidate_chars: int
    char_ratio: float
    sequence_ratio: float
    jaccard_words: float
    code_fence_delta: int
    html_table_delta: int
    html_div_delta: int
    markdown_table_row_delta: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "page": self.page,
            "baseline_chars": self.baseline_chars,
            "candidate_chars": self.candidate_chars,
            "char_ratio": round(self.char_ratio, 6),
            "sequence_ratio": round(self.sequence_ratio, 6),
            "jaccard_words": round(self.jaccard_words, 6),
            "code_fence_delta": self.code_fence_delta,
            "html_table_delta": self.html_table_delta,
            "html_div_delta": self.html_div_delta,
            "markdown_table_row_delta": self.markdown_table_row_delta,
        }


def _read_pages(directory: Path) -> Dict[str, str]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    pages: Dict[str, str] = {}
    for path in sorted(directory.glob("*.md")):
        pages[path.name] = path.read_text(encoding="utf-8", errors="replace")
    if not pages:
        raise ValueError(f"No markdown pages found under: {directory}")
    return pages


def _word_jaccard(a: str, b: str) -> float:
    words_a = set(WORD_RE.findall(a.lower()))
    words_b = set(WORD_RE.findall(b.lower()))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a.intersection(words_b))
    union = len(words_a.union(words_b))
    return overlap / float(union) if union else 1.0


def _sequence_ratio(a: str, b: str) -> float:
    # Using Python stdlib keeps this script dependency-light.
    import difflib

    return difflib.SequenceMatcher(a=a, b=b).ratio()


def _count_markdown_table_rows(text: str) -> int:
    lines = text.splitlines()
    count = 0
    for line in lines:
        if "|" in line and len(line.strip()) >= 3:
            count += 1
    return count


def _page_metrics(page: str, baseline: str, candidate: str) -> PageMetrics:
    baseline_chars = len(baseline)
    candidate_chars = len(candidate)
    char_ratio = (candidate_chars / baseline_chars) if baseline_chars else 1.0
    return PageMetrics(
        page=page,
        baseline_chars=baseline_chars,
        candidate_chars=candidate_chars,
        char_ratio=char_ratio,
        sequence_ratio=_sequence_ratio(baseline, candidate),
        jaccard_words=_word_jaccard(baseline, candidate),
        code_fence_delta=abs(baseline.count("```") - candidate.count("```")),
        html_table_delta=abs(
            (baseline.count("<table") - baseline.count("</table>"))
            - (candidate.count("<table") - candidate.count("</table>"))
        ),
        html_div_delta=abs(
            (baseline.count("<div") - baseline.count("</div>"))
            - (candidate.count("<div") - candidate.count("</div>"))
        ),
        markdown_table_row_delta=abs(
            _count_markdown_table_rows(baseline) - _count_markdown_table_rows(candidate)
        ),
    )


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * pct / 100.0))))
    return ordered[rank]


def _aggregate(results: List[PageMetrics]) -> Dict[str, float]:
    sequence = [m.sequence_ratio for m in results]
    jaccard = [m.jaccard_words for m in results]
    char_ratio = [m.char_ratio for m in results]
    structure_penalty = [
        (m.code_fence_delta + m.html_table_delta + m.html_div_delta + m.markdown_table_row_delta)
        for m in results
    ]
    return {
        "pages": len(results),
        "sequence_ratio_mean": round(statistics.mean(sequence), 6) if sequence else 0.0,
        "sequence_ratio_p10": round(_percentile(sequence, 10.0), 6),
        "jaccard_words_mean": round(statistics.mean(jaccard), 6) if jaccard else 0.0,
        "jaccard_words_p10": round(_percentile(jaccard, 10.0), 6),
        "char_ratio_mean": round(statistics.mean(char_ratio), 6) if char_ratio else 1.0,
        "char_ratio_p10": round(_percentile(char_ratio, 10.0), 6),
        "char_ratio_p90": round(_percentile(char_ratio, 90.0), 6),
        "structure_penalty_mean": round(statistics.mean(structure_penalty), 6) if structure_penalty else 0.0,
        "structure_penalty_max": max(structure_penalty) if structure_penalty else 0,
    }


def _recommendation(agg: Dict[str, float]) -> Dict[str, str]:
    # Conservative heuristic gates for quick triage.
    seq = agg["sequence_ratio_mean"]
    jac = agg["jaccard_words_mean"]
    p10 = agg["char_ratio_p10"]
    p90 = agg["char_ratio_p90"]
    structure = agg["structure_penalty_max"]
    if seq >= 0.84 and jac >= 0.78 and p10 >= 0.65 and p90 <= 1.45 and structure <= 12:
        verdict = "pass"
        note = "Candidate is close enough to baseline for default traffic with manual spot-check."
    elif seq >= 0.75 and jac >= 0.68 and p10 >= 0.55 and structure <= 24:
        verdict = "warn"
        note = "Candidate is likely usable for latency-priority mode but not as default."
    else:
        verdict = "fail"
        note = "Candidate diverges too much from baseline; keep behind opt-in only."
    return {"verdict": verdict, "note": note}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Chandra per-page markdown outputs.")
    parser.add_argument("--baseline-dir", required=True, help="Directory of baseline per-page markdown files.")
    parser.add_argument("--candidate-dir", required=True, help="Directory of candidate per-page markdown files.")
    parser.add_argument("--out-json", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    baseline_pages = _read_pages(Path(args.baseline_dir))
    candidate_pages = _read_pages(Path(args.candidate_dir))

    shared_pages = sorted(set(baseline_pages).intersection(candidate_pages))
    if not shared_pages:
        raise ValueError("No overlapping page filenames between baseline and candidate directories.")

    missing_from_candidate = sorted(set(baseline_pages).difference(candidate_pages))
    missing_from_baseline = sorted(set(candidate_pages).difference(baseline_pages))

    rows: List[PageMetrics] = []
    for page in shared_pages:
        rows.append(_page_metrics(page, baseline_pages[page], candidate_pages[page]))

    aggregate = _aggregate(rows)
    recommendation = _recommendation(aggregate)
    report = {
        "baseline_dir": str(Path(args.baseline_dir)),
        "candidate_dir": str(Path(args.candidate_dir)),
        "shared_pages": len(shared_pages),
        "missing_from_candidate": missing_from_candidate,
        "missing_from_baseline": missing_from_baseline,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "pages": [row.as_dict() for row in rows],
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
