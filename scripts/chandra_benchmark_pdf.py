#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
from PIL import Image

from QueryLake.operation_classes.ray_chandra_class import ChandraDeployment


@dataclass
class PassStats:
    name: str
    wall_seconds: float
    page_latencies: List[float]
    total_chars: int
    outputs: List[str]
    metadata: Optional[dict] = None

    def as_dict(self) -> dict:
        latencies_ms = [value * 1000.0 for value in self.page_latencies]
        pages = len(self.page_latencies)
        result = {
            "name": self.name,
            "pages": pages,
            "wall_seconds": round(self.wall_seconds, 4),
            "pages_per_second": round((pages / self.wall_seconds), 4) if self.wall_seconds > 0 else 0.0,
            "total_chars": self.total_chars,
            "latency_ms": {
                "mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
                "median": round(statistics.median(latencies_ms), 2) if latencies_ms else 0.0,
                "p95": round(_percentile(latencies_ms, 95.0), 2) if latencies_ms else 0.0,
                "max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
            },
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * (pct / 100.0))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _resolve_pdf_path(pdf_path: str) -> Tuple[str, Optional[str]]:
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        fd, tmp_path = tempfile.mkstemp(prefix="chandra_bench_", suffix=".pdf")
        os.close(fd)
        urllib.request.urlretrieve(pdf_path, tmp_path)
        return tmp_path, tmp_path
    return pdf_path, None


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _render_pdf_pages(
    pdf_path: str,
    dpi: int,
    max_pages: Optional[int],
    render_cache_dir: Optional[str] = None,
) -> List[Any]:
    document = pdfium.PdfDocument(pdf_path)
    scale = float(dpi) / 72.0
    page_count = len(document)
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    cache_run_dir: Optional[Path] = None
    if render_cache_dir:
        cache_root = Path(render_cache_dir)
        cache_key = f"{_hash_file(pdf_path)}_dpi{dpi}_pages{page_count}"
        cache_run_dir = cache_root / cache_key
        cache_run_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Any] = []
    for index in range(page_count):
        cache_file = cache_run_dir / f"page_{index+1:04d}.png" if cache_run_dir else None
        if cache_file and cache_file.exists():
            pil_image = Image.open(cache_file).convert("RGB")
            pages.append(pil_image)
            continue

        page = document[index]
        pil_image = page.render(scale=scale).to_pil().convert("RGB")
        pages.append(pil_image)
        if cache_file:
            pil_image.save(cache_file, format="PNG")
        page.close()
    document.close()
    return pages


def _resolve_runtime_class():
    runtime_class = getattr(ChandraDeployment, "func_or_class", None)
    return runtime_class or ChandraDeployment


async def _run_pass(
    runtime: Any,
    images: List[Any],
    prompt: str,
    max_new_tokens: Optional[int],
    max_image_pixels: Optional[int],
    concurrency: int,
    name: str,
    profile: Optional[str] = None,
) -> PassStats:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    page_latencies: List[float] = [0.0] * len(images)
    outputs: List[str] = [""] * len(images)

    async def _worker(page_idx: int, image_obj: Any) -> None:
        start = time.perf_counter()
        async with semaphore:
            output = await runtime.transcribe(
                image=image_obj,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                max_image_pixels=max_image_pixels,
                profile=profile,
            )
        page_latencies[page_idx] = time.perf_counter() - start
        outputs[page_idx] = output

    wall_start = time.perf_counter()
    await asyncio.gather(*[_worker(page_idx, image) for page_idx, image in enumerate(images)])
    wall_seconds = time.perf_counter() - wall_start
    return PassStats(
        name=name,
        wall_seconds=wall_seconds,
        page_latencies=page_latencies,
        total_chars=sum(len(text) for text in outputs),
        outputs=outputs,
    )


async def _run_subset_pass(
    runtime: Any,
    images: List[Any],
    indices: List[int],
    prompt: str,
    max_new_tokens: Optional[int],
    max_image_pixels: Optional[int],
    concurrency: int,
    profile: Optional[str] = None,
) -> Tuple[float, Dict[int, float], Dict[int, str]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    page_latencies: Dict[int, float] = {}
    page_outputs: Dict[int, str] = {}

    async def _worker(page_idx: int) -> None:
        start = time.perf_counter()
        async with semaphore:
            output = await runtime.transcribe(
                image=images[page_idx],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                max_image_pixels=max_image_pixels,
                profile=profile,
            )
        page_latencies[page_idx] = time.perf_counter() - start
        page_outputs[page_idx] = output

    wall_start = time.perf_counter()
    await asyncio.gather(*[_worker(page_idx) for page_idx in indices])
    wall_seconds = time.perf_counter() - wall_start
    return wall_seconds, page_latencies, page_outputs


def _needs_escalation(text: str, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < max(1, int(min_chars)):
        return True
    if stripped.count("```") % 2 == 1:
        return True
    if stripped.count("<table") > stripped.count("</table>"):
        return True
    if stripped.count("<div") > stripped.count("</div>"):
        return True
    if stripped.count("\\begin{") > stripped.count("\\end{"):
        return True
    suspect_tail = ("<table", "<tr", "<td", "<th", "<div", "<span", "\\begin{", "| --- |")
    if any(stripped.endswith(suffix) for suffix in suspect_tail):
        return True
    return False


async def _run_adaptive_pass(
    runtime: Any,
    images: List[Any],
    prompt: str,
    concurrency: int,
    low_tokens: int,
    high_tokens: int,
    low_pixels: Optional[int],
    high_pixels: Optional[int],
    min_chars: int,
    name: str,
    profile: Optional[str] = None,
) -> PassStats:
    low_pass = await _run_pass(
        runtime=runtime,
        images=images,
        prompt=prompt,
            max_new_tokens=low_tokens,
        max_image_pixels=low_pixels,
        concurrency=concurrency,
        name=f"{name}_low",
        profile=profile,
    )
    flagged = [idx for idx, output in enumerate(low_pass.outputs) if _needs_escalation(output, min_chars=min_chars)]
    merged_outputs = list(low_pass.outputs)
    merged_latencies = list(low_pass.page_latencies)
    total_wall = low_pass.wall_seconds
    rerun_wall = 0.0

    if flagged:
        rerun_wall, rerun_latencies, rerun_outputs = await _run_subset_pass(
            runtime=runtime,
            images=images,
            indices=flagged,
            prompt=prompt,
            max_new_tokens=high_tokens,
            max_image_pixels=high_pixels,
            concurrency=concurrency,
            profile=profile,
        )
        total_wall += rerun_wall
        for idx in flagged:
            merged_outputs[idx] = rerun_outputs[idx]
            merged_latencies[idx] += rerun_latencies[idx]

    return PassStats(
        name=name,
        wall_seconds=total_wall,
        page_latencies=merged_latencies,
        total_chars=sum(len(text) for text in merged_outputs),
        outputs=merged_outputs,
        metadata={
            "adaptive": True,
            "low_tokens": low_tokens,
            "high_tokens": high_tokens,
            "low_pixels": low_pixels,
            "high_pixels": high_pixels,
            "flagged_pages": len(flagged),
            "flagged_indices": flagged,
            "rerun_wall_seconds": round(rerun_wall, 4),
        },
    )


async def _run(args: argparse.Namespace) -> dict:
    resolved_pdf_path, temporary_download = _resolve_pdf_path(args.pdf)
    try:
        render_start = time.perf_counter()
        images = _render_pdf_pages(
            resolved_pdf_path,
            dpi=args.dpi,
            max_pages=args.max_pages,
            render_cache_dir=args.render_cache_dir,
        )
        render_seconds = time.perf_counter() - render_start

        runtime_class = _resolve_runtime_class()
        load_start = time.perf_counter()
        runtime = runtime_class(
            model_path=args.model,
            prompt=args.prompt,
            max_batch_size=args.max_batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            cache_enabled=not args.disable_cache,
            torch_dtype=args.torch_dtype,
            runtime_backend=args.runtime_backend,
            vllm_trust_remote_code=args.vllm_trust_remote_code,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_data_parallel_size=args.vllm_data_parallel_size,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_max_num_seqs=args.vllm_max_num_seqs,
            vllm_dtype=args.vllm_dtype,
            vllm_enforce_eager=args.vllm_enforce_eager,
            vllm_disable_log_stats=args.vllm_disable_log_stats,
            vllm_async_scheduling=args.vllm_async_scheduling,
            vllm_server_base_url=args.vllm_server_base_url,
            vllm_server_base_urls=args.vllm_server_base_urls,
            vllm_server_model=args.vllm_server_model,
            vllm_server_api_key=args.vllm_server_api_key,
            vllm_server_timeout_seconds=args.vllm_server_timeout_seconds,
            vllm_server_max_retries=args.vllm_server_max_retries,
            vllm_server_retry_backoff_seconds=args.vllm_server_retry_backoff_seconds,
            vllm_server_parallel_requests=args.vllm_server_parallel_requests,
            vllm_server_probe_on_init=args.vllm_server_probe_on_init,
            vllm_server_fallback_to_hf_on_error=args.vllm_server_fallback_to_hf_on_error,
            vllm_server_circuit_breaker_threshold=args.vllm_server_circuit_breaker_threshold,
        )
        model_load_seconds = time.perf_counter() - load_start

        passes: List[PassStats] = []

        async def _execute_pass(pass_name: str) -> PassStats:
            resolved_max_new_tokens = None if args.use_profile_defaults else args.max_new_tokens
            resolved_max_image_pixels = None if args.use_profile_defaults else args.max_image_pixels
            if args.adaptive:
                return await _run_adaptive_pass(
                    runtime=runtime,
                    images=images,
                    prompt=args.prompt,
                    concurrency=args.concurrency,
                    low_tokens=args.adaptive_low_tokens,
                    high_tokens=args.adaptive_high_tokens,
                    low_pixels=args.adaptive_low_pixels,
                    high_pixels=args.adaptive_high_pixels,
                    min_chars=args.adaptive_min_chars,
                    name=pass_name,
                    profile=args.profile,
                )
            return await _run_pass(
                runtime=runtime,
                images=images,
                prompt=args.prompt,
                max_new_tokens=resolved_max_new_tokens,
                max_image_pixels=resolved_max_image_pixels,
                concurrency=args.concurrency,
                name=pass_name,
                profile=args.profile,
            )
        passes.append(await _execute_pass("cold"))

        for warm_idx in range(args.warm_runs):
            passes.append(await _execute_pass(f"warm_{warm_idx + 1}"))

        if getattr(runtime, "_batcher", None) is not None:
            await runtime._batcher.shutdown()

        if args.out_markdown:
            pass_name = args.markdown_pass
            selected_pass = passes[-1]
            if pass_name != "last":
                matched = next((item for item in passes if item.name == pass_name), None)
                if matched is None:
                    available = ", ".join(item.name for item in passes)
                    raise ValueError(f"Unknown --markdown-pass '{pass_name}'. Available: {available}")
                selected_pass = matched
            markdown_text = []
            for page_idx, page_text in enumerate(selected_pass.outputs, start=1):
                markdown_text.append(f"## Page {page_idx}\n\n{page_text.strip()}\n")
            out_md = Path(args.out_markdown)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("\n---\n\n".join(markdown_text) + "\n", encoding="utf-8")

        if args.out_pages_dir:
            pages_dir = Path(args.out_pages_dir)
            pages_dir.mkdir(parents=True, exist_ok=True)
            for page_idx, page_text in enumerate(passes[-1].outputs, start=1):
                page_path = pages_dir / f"page_{page_idx:04d}.md"
                page_path.write_text(page_text.strip() + "\n", encoding="utf-8")

        results = {
            "pdf": args.pdf,
            "render": {
                "dpi": args.dpi,
                "pages": len(images),
                "seconds": round(render_seconds, 4),
                "cache_dir": args.render_cache_dir,
            },
            "runtime": {
                "model": args.model,
                "torch_dtype": args.torch_dtype,
                "max_batch_size": args.max_batch_size,
                "max_batch_wait_ms": args.max_batch_wait_ms,
                "max_new_tokens": None if args.use_profile_defaults else args.max_new_tokens,
                "max_image_pixels": None if args.use_profile_defaults else args.max_image_pixels,
                "concurrency": args.concurrency,
                "model_load_seconds": round(model_load_seconds, 4),
                "profile": args.profile,
                "runtime_backend": args.runtime_backend,
                "vllm_server_base_urls": args.vllm_server_base_urls,
                "use_profile_defaults": bool(args.use_profile_defaults),
                "cache_enabled": not args.disable_cache,
                "adaptive": bool(args.adaptive),
                "adaptive_low_tokens": args.adaptive_low_tokens,
                "adaptive_high_tokens": args.adaptive_high_tokens,
                "adaptive_low_pixels": args.adaptive_low_pixels,
                "adaptive_high_pixels": args.adaptive_high_pixels,
                "adaptive_min_chars": args.adaptive_min_chars,
            },
            "passes": [item.as_dict() for item in passes],
        }
        return results
    finally:
        if temporary_download and os.path.exists(temporary_download):
            os.remove(temporary_download)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Chandra OCR on a PDF.")
    parser.add_argument("--pdf", required=True, help="Local PDF path or http(s) URL.")
    parser.add_argument("--model", required=True, help="Model path (HF id or local path).")
    parser.add_argument("--prompt", default="Convert this page into clean Markdown.", help="OCR prompt.")
    parser.add_argument("--dpi", type=int, default=144, help="Render DPI for PDF pages.")
    parser.add_argument("--render-cache-dir", default=None, help="Optional on-disk cache directory for rendered pages.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page cap.")
    parser.add_argument("--concurrency", type=int, default=24, help="Concurrent page requests.")
    parser.add_argument("--warm-runs", type=int, default=1, help="Number of warm passes after cold pass.")
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--max-batch-wait-ms", type=int, default=15)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Deprecated runtime knob; Chandra runtime uses a fixed output token budget.",
    )
    parser.add_argument("--max-image-pixels", type=int, default=2_097_152)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--runtime-backend", default="hf", choices=["hf", "vllm", "vllm_server"])
    parser.add_argument("--vllm-trust-remote-code", action="store_true")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--vllm-max-model-len", type=int, default=131072)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-disable-log-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-async-scheduling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-server-base-url", default="http://127.0.0.1:8022/v1")
    parser.add_argument(
        "--vllm-server-base-urls",
        default="",
        help="Optional comma-separated vLLM server /v1 base URLs for striped dispatch.",
    )
    parser.add_argument("--vllm-server-model", default=None)
    parser.add_argument("--vllm-server-api-key", default="")
    parser.add_argument("--vllm-server-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--vllm-server-max-retries", type=int, default=2)
    parser.add_argument("--vllm-server-retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument("--vllm-server-parallel-requests", type=int, default=24)
    parser.add_argument("--vllm-server-probe-on-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-server-fallback-to-hf-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-server-circuit-breaker-threshold", type=int, default=3)
    parser.add_argument("--profile", default=None, help="Optional runtime profile (speed|balanced|quality|adaptive).")
    parser.add_argument("--disable-cache", action="store_true", help="Disable runtime in-memory cache for benchmark run.")
    parser.add_argument(
        "--use-profile-defaults",
        action="store_true",
        help="Do not override tokens/pixels; let runtime profile/default choose them.",
    )
    parser.add_argument("--adaptive", action="store_true", help="Enable two-stage low/high rerun policy.")
    parser.add_argument(
        "--adaptive-low-tokens",
        type=int,
        default=512,
        help="Deprecated runtime knob; retained for benchmark script compatibility.",
    )
    parser.add_argument(
        "--adaptive-high-tokens",
        type=int,
        default=1024,
        help="Deprecated runtime knob; retained for benchmark script compatibility.",
    )
    parser.add_argument("--adaptive-low-pixels", type=int, default=1_048_576)
    parser.add_argument("--adaptive-high-pixels", type=int, default=2_097_152)
    parser.add_argument("--adaptive-min-chars", type=int, default=450)
    parser.add_argument("--out-json", default=None, help="Optional path to write JSON results.")
    parser.add_argument("--out-markdown", default=None, help="Optional path to write merged markdown.")
    parser.add_argument("--out-pages-dir", default=None, help="Optional directory for per-page markdown from last pass.")
    parser.add_argument(
        "--markdown-pass",
        default="last",
        help="Pass name to export for --out-markdown (default: last).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
