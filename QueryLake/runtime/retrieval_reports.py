from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _aggregate_numeric(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums.keys()) if counts[key] > 0}


def build_experiment_report(
    *,
    experiment_id: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    baseline_avg = _aggregate_numeric(baseline_rows)
    candidate_avg = _aggregate_numeric(candidate_rows)
    delta_avg = _aggregate_numeric(delta_rows)

    report_json = {
        "experiment_id": experiment_id,
        "row_count": len(delta_rows),
        "baseline_avg": baseline_avg,
        "candidate_avg": candidate_avg,
        "delta_avg": delta_avg,
    }

    lines = [
        f"# Retrieval Experiment Report: {experiment_id}",
        "",
        f"- Rows compared: {len(delta_rows)}",
        "",
        "## Baseline Averages",
    ]
    for key, value in baseline_avg.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    lines.append("## Candidate Averages")
    for key, value in candidate_avg.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    lines.append("## Delta Averages (candidate - baseline)")
    for key, value in delta_avg.items():
        lines.append(f"- {key}: {value:.6f}")

    return report_json, "\n".join(lines) + "\n"
