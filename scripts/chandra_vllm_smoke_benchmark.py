#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List


def _render_pages(pdf_path: Path, max_pages: int) -> List[Any]:
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(pdf_path))
    count = min(len(pdf), max_pages)
    pages: List[Any] = []
    for i in range(count):
        page = pdf[i]
        pages.append(page.render(scale=2.0).to_pil().convert("RGB"))
        page.close()
    pdf.close()
    return pages


def _run_page(llm: Any, image: Any, prompt: str, max_tokens: int) -> str:
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_pil", "image_pil": image},
            ],
        }
    ]
    outputs = llm.chat(messages=messages, sampling_params=params)
    if outputs and hasattr(outputs[0], "outputs") and outputs[0].outputs:
        return outputs[0].outputs[0].text
    return str(outputs[0]) if outputs else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple Chandra-on-vLLM smoke benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--max-pages", type=int, required=True)
    parser.add_argument("--prompt", default="Convert this page into clean Markdown.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--out-json", default=None)
    parser.add_argument(
        "--out-md",
        default=None,
        help="Optional path to write merged Markdown output for all processed pages.",
    )
    parser.add_argument(
        "--out-pages-dir",
        default=None,
        help="Optional directory to write per-page Markdown outputs (page_0001.md, ...).",
    )
    args = parser.parse_args()

    from vllm import LLM  # type: ignore

    timings: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    start = time.perf_counter()
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    timings["load_seconds"] = round(time.perf_counter() - load_start, 4)

    render_start = time.perf_counter()
    images = _render_pages(Path(args.pdf), args.max_pages)
    timings["render_seconds"] = round(time.perf_counter() - render_start, 4)

    page_latencies: List[float] = []
    page_chars: List[int] = []
    page_outputs: List[str] = []
    infer_start = time.perf_counter()
    for image in images:
        t0 = time.perf_counter()
        out = _run_page(llm, image, args.prompt, args.max_tokens)
        page_latencies.append(time.perf_counter() - t0)
        page_chars.append(len(out))
        page_outputs.append(out)
    timings["infer_seconds"] = round(time.perf_counter() - infer_start, 4)
    timings["total_seconds"] = round(time.perf_counter() - start, 4)
    timings["pages"] = len(images)
    timings["page_latency_seconds"] = {
        "mean": round(statistics.mean(page_latencies), 4) if page_latencies else 0.0,
        "median": round(statistics.median(page_latencies), 4) if page_latencies else 0.0,
        "max": round(max(page_latencies), 4) if page_latencies else 0.0,
    }
    timings["pages_per_second_infer"] = (
        round((len(images) / max(1e-9, sum(page_latencies))), 4) if page_latencies else 0.0
    )
    timings["total_chars"] = int(sum(page_chars))
    timings["input"] = {
        "model": args.model,
        "pdf": args.pdf,
        "max_pages": args.max_pages,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
    }

    output = json.dumps(timings, indent=2)
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(output, encoding="utf-8")

    if args.out_pages_dir:
        pages_dir = Path(args.out_pages_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)
        for idx, page_md in enumerate(page_outputs, start=1):
            page_path = pages_dir / f"page_{idx:04d}.md"
            page_path.write_text(page_md.strip() + "\n", encoding="utf-8")

    if args.out_md:
        merged = []
        for idx, page_md in enumerate(page_outputs, start=1):
            merged.append(f"<!-- page:{idx} -->")
            merged.append("")
            merged.append(page_md.strip())
            merged.append("")
        merged_path = Path(args.out_md)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text("\n".join(merged).strip() + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
