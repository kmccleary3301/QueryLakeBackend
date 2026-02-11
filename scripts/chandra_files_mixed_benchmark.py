#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from QueryLake.files.object_store import LocalCASObjectStore
from QueryLake.files.service import FilesRuntimeService
from QueryLake.operation_classes.ray_chandra_class import ChandraDeployment


class _DummyDB:
    def add(self, _row: Any) -> None:
        return None

    def commit(self) -> None:
        return None

    def exec(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def get(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        return None


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, **kwargs):
        return await self._fn(**kwargs)


class _LocalChandraHandle:
    def __init__(self, deployment_impl):
        self.transcribe = _RemoteMethod(deployment_impl.transcribe)


class _DummyUmbrella:
    def __init__(self, handle):
        self.chandra_handles = {"chandra": handle}


@dataclass
class RunResult:
    wall_seconds: float
    meta: Dict[str, Any]
    extra: Dict[str, Any]


def _parse_base_urls(raw: str) -> List[str]:
    urls = [value.strip() for value in (raw or "").split(",") if value.strip()]
    if not urls:
        raise ValueError("Provide at least one vLLM base URL.")
    return urls


def _make_files_service(
    *,
    object_store_dir: str,
    umbrella: Any,
    text_layer_mode: str,
    min_chars_per_page: int,
    min_coverage: float,
) -> FilesRuntimeService:
    os.environ["QUERYLAKE_PDF_TEXT_LAYER_MODE"] = text_layer_mode
    os.environ["QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE"] = str(int(min_chars_per_page))
    os.environ["QUERYLAKE_PDF_TEXT_MIN_COVERAGE"] = str(float(min_coverage))
    return FilesRuntimeService(_DummyDB(), object_store=LocalCASObjectStore(base_dir=object_store_dir), umbrella=umbrella)


async def _run_full_ocr(
    *,
    pdf_bytes: bytes,
    profile: str,
    dpi: int,
    concurrency: int,
    chandra_handle: Any,
    text_layer_mode: str,
    min_chars_per_page: int,
    min_coverage: float,
) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="ql_chandra_files_full_") as tmp_dir:
        service = _make_files_service(
            object_store_dir=tmp_dir,
            umbrella=_DummyUmbrella(chandra_handle),
            text_layer_mode=text_layer_mode,
            min_chars_per_page=min_chars_per_page,
            min_coverage=min_coverage,
        )
        bytes_cas = service.store.put_bytes(pdf_bytes)
        start = time.perf_counter()
        _text, meta = await service._process_pdf_with_chandra(
            pdf_bytes,
            bytes_cas,
            profile=profile,
            dpi=dpi,
            concurrency=concurrency,
        )
        wall = time.perf_counter() - start
        return RunResult(wall_seconds=wall, meta=meta, extra={})


async def _run_mixed(
    *,
    pdf_bytes: bytes,
    profile: str,
    dpi: int,
    concurrency: int,
    chandra_handle: Any,
    min_chars_per_page: int,
    min_coverage: float,
) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="ql_chandra_files_mixed_") as tmp_dir:
        service = _make_files_service(
            object_store_dir=tmp_dir,
            umbrella=_DummyUmbrella(chandra_handle),
            text_layer_mode="mixed",
            min_chars_per_page=min_chars_per_page,
            min_coverage=min_coverage,
        )
        bytes_cas = service.store.put_bytes(pdf_bytes)

        extract_start = time.perf_counter()
        page_texts, parsed_meta = service._extract_pdf_text_layer_pages(pdf_bytes)
        extract_seconds = time.perf_counter() - extract_start

        overrides: Dict[int, str] = {}
        routing_meta: Dict[str, Any] = {}
        if page_texts is not None:
            routing = service._evaluate_pdf_text_layer_page_overrides(page_texts)
            routing_meta = {key: value for key, value in routing.items() if key != "selected_page_indices"}
            indices = list(routing.get("selected_page_indices", []))
            overrides = {int(idx): page_texts[int(idx)] for idx in indices}

        ocr_start = time.perf_counter()
        _text, meta = await service._process_pdf_with_chandra(
            pdf_bytes,
            bytes_cas,
            profile=profile,
            dpi=dpi,
            concurrency=concurrency,
            page_text_overrides=overrides,
        )
        ocr_seconds = time.perf_counter() - ocr_start
        return RunResult(
            wall_seconds=extract_seconds + ocr_seconds,
            meta=meta,
            extra={
                "text_layer_extract_seconds": round(extract_seconds, 4),
                "text_layer_parsed_meta": parsed_meta,
                "text_layer_routing": routing_meta,
                "text_layer_selected_pages": len(overrides),
            },
        )


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    pdf_path = Path(args.pdf)
    pdf_bytes = pdf_path.read_bytes()

    base_urls = _parse_base_urls(args.vllm_base_urls)
    impl = ChandraDeployment.func_or_class(
        model_path=args.model,
        runtime_backend="vllm_server",
        vllm_server_base_urls=base_urls,
        vllm_server_model=args.vllm_model,
        vllm_server_api_key=args.vllm_api_key,
        vllm_server_parallel_requests=int(args.vllm_parallel_requests),
        vllm_server_probe_on_init=True,
        vllm_server_fallback_to_hf_on_error=False,
        cache_enabled=False,
    )
    chandra_handle = _LocalChandraHandle(impl)

    full = await _run_full_ocr(
        pdf_bytes=pdf_bytes,
        profile=args.profile,
        dpi=args.dpi,
        concurrency=args.concurrency,
        chandra_handle=chandra_handle,
        text_layer_mode="off",
        min_chars_per_page=args.min_chars_per_page,
        min_coverage=args.min_coverage,
    )
    mixed = await _run_mixed(
        pdf_bytes=pdf_bytes,
        profile=args.profile,
        dpi=args.dpi,
        concurrency=args.concurrency,
        chandra_handle=chandra_handle,
        min_chars_per_page=args.min_chars_per_page,
        min_coverage=args.min_coverage,
    )

    result = {
        "pdf": str(pdf_path),
        "profile": args.profile,
        "dpi": args.dpi,
        "concurrency": args.concurrency,
        "vllm_base_urls": base_urls,
        "full_ocr": {
            "wall_seconds": round(full.wall_seconds, 4),
            "meta": full.meta,
        },
        "mixed": {
            "wall_seconds": round(mixed.wall_seconds, 4),
            "meta": mixed.meta,
            "extra": mixed.extra,
        },
    }
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Files mixed text-layer + Chandra page routing.")
    parser.add_argument("--pdf", required=True, help="Local PDF path.")
    parser.add_argument("--model", default="models/chandra", help="Chandra model path.")
    parser.add_argument("--profile", default="speed", help="Chandra profile (speed|balanced|quality|adaptive).")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--min-chars-per-page", type=int, default=120)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--vllm-base-urls", default="http://127.0.0.1:8022/v1,http://127.0.0.1:8023/v1")
    parser.add_argument("--vllm-model", default="chandra")
    parser.add_argument("--vllm-api-key", default="chandra-local-key")
    parser.add_argument("--vllm-parallel-requests", type=int, default=24)
    parser.add_argument("--out-json", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = asyncio.run(_run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
