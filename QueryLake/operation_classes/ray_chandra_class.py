from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import io
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from ray import serve

logger = logging.getLogger(__name__)
FIXED_OUTPUT_MAX_NEW_TOKENS = 768


@dataclass(frozen=True)
class ChandraRequest:
    image: Any
    prompt: str
    max_new_tokens: int
    max_image_pixels: Optional[int] = None
    cache_key: Optional[str] = None
    allow_escalation: bool = False
    escalation_max_new_tokens: Optional[int] = None
    escalation_max_image_pixels: Optional[int] = None
    adaptive_rerun_limit: Optional[int] = None


class _MicroBatcher:
    def __init__(self, process_fn, max_batch_size: int, max_batch_wait_ms: int) -> None:
        self._process_fn = process_fn
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_batch_wait_ms = max(1, int(max_batch_wait_ms))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def submit(self, item: ChandraRequest) -> str:
        if self._worker_task is None:
            self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((item, future))
        return await future

    async def shutdown(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def _worker_loop(self) -> None:
        try:
            while True:
                item, future = await self._queue.get()
                batch_items = [(item, future)]
                batch_start = asyncio.get_running_loop().time()
                while len(batch_items) < self._max_batch_size:
                    remaining = (self._max_batch_wait_ms / 1000.0) - (
                        asyncio.get_running_loop().time() - batch_start
                    )
                    if remaining <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    batch_items.append(next_item)
                try:
                    requests = [entry[0] for entry in batch_items]
                    results = await self._process_fn(requests)
                    if len(results) != len(batch_items):
                        raise RuntimeError("Batch size mismatch in Chandra processing")
                    for (_, future), result in zip(batch_items, results):
                        if not future.done():
                            future.set_result(result)
                except Exception as exc:
                    for _, future in batch_items:
                        if not future.done():
                            future.set_exception(exc)
        except asyncio.CancelledError:
            return


@serve.deployment
class ChandraDeployment:
    """
    Chandra OCR runtime with four profile modes:
    - speed: lowest latency, higher truncation risk
    - balanced: default production profile
    - quality: higher fidelity, slower
    - adaptive: speed-first with bounded rerun escalation for flagged outputs

    Note: output token budget is fixed globally to avoid using truncation-prone token caps
    as a tuning lever. Profile differences come from image policy + escalation only.
    """

    def __init__(
        self,
        model_path: str,
        prompt: str = (
            "You are an OCR system. Convert the page image to Markdown, preserving reading order.\n"
            "Output Markdown only (no HTML, no XML, no code fences).\n"
            "- Use Markdown headings, lists, and tables.\n"
            "- For tables, use GitHub-flavored Markdown table syntax.\n"
            "- Preserve the text exactly; do not add commentary.\n"
            "- If a character is illegible, use '?'."
        ),
        max_batch_size: int = 24,
        max_batch_wait_ms: int = 15,
        max_new_tokens: int = 768,
        max_image_pixels: Optional[int] = 1_048_576,
        size_bucket_ratio: float = 1.8,
        enable_timing_logs: bool = True,
        default_profile: str = "balanced",
        profile_speed_max_new_tokens: int = 512,
        profile_speed_max_image_pixels: int = 1_048_576,
        profile_balanced_max_new_tokens: int = 768,
        profile_balanced_max_image_pixels: int = 1_048_576,
        profile_quality_max_new_tokens: int = 1024,
        profile_quality_max_image_pixels: int = 2_097_152,
        adaptive_high_max_new_tokens: int = 1024,
        adaptive_high_max_image_pixels: int = 2_097_152,
        adaptive_min_chars: int = 450,
        adaptive_max_reruns_per_batch: int = 4,
        adaptive_max_batch_seconds: float = 60.0,
        cache_enabled: bool = True,
        cache_max_entries: int = 512,
        torch_dtype: Optional[str] = "auto",
        runtime_backend: str = "hf",
        vllm_trust_remote_code: bool = False,
        vllm_tensor_parallel_size: int = 1,
        vllm_data_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.95,
        vllm_max_model_len: int = 131072,
        vllm_max_num_seqs: int = 8,
        vllm_dtype: str = "auto",
        vllm_enforce_eager: bool = False,
        vllm_disable_log_stats: bool = True,
        vllm_mm_encoder_tp_mode: Optional[str] = None,
        vllm_async_scheduling: bool = False,
        vllm_enable_chunked_prefill: Optional[bool] = None,
        vllm_fallback_to_hf_on_error: bool = True,
        vllm_server_base_url: str = "http://127.0.0.1:8022/v1",
        vllm_server_base_urls: Optional[List[str]] = None,
        vllm_server_model: Optional[str] = None,
        vllm_server_api_key: Optional[str] = None,
        vllm_server_timeout_seconds: float = 120.0,
        vllm_server_max_retries: int = 2,
        vllm_server_retry_backoff_seconds: float = 0.5,
        vllm_server_parallel_requests: int = 24,
        vllm_server_probe_on_init: bool = True,
        vllm_server_fallback_to_hf_on_error: bool = True,
        vllm_server_circuit_breaker_threshold: int = 3,
    ) -> None:
        self._model_path = model_path
        self._prompt = prompt
        self._fixed_max_new_tokens = int(FIXED_OUTPUT_MAX_NEW_TOKENS)
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_ms = int(max_batch_wait_ms)
        self._max_image_pixels = int(max_image_pixels) if max_image_pixels else None
        self._size_bucket_ratio = max(1.1, float(size_bucket_ratio))
        self._enable_timing_logs = bool(enable_timing_logs)
        self._default_profile = str(default_profile or "balanced").strip().lower()
        self._profile_settings = {
            "speed": {
                "max_image_pixels": int(profile_speed_max_image_pixels),
            },
            "balanced": {
                "max_image_pixels": int(profile_balanced_max_image_pixels),
            },
            "quality": {
                "max_image_pixels": int(profile_quality_max_image_pixels),
            },
        }
        self._adaptive_high_max_new_tokens = int(self._fixed_max_new_tokens)
        self._adaptive_high_max_image_pixels = int(adaptive_high_max_image_pixels)
        self._adaptive_min_chars = max(1, int(adaptive_min_chars))
        self._adaptive_max_reruns_per_batch = max(0, int(adaptive_max_reruns_per_batch))
        self._adaptive_max_batch_seconds = max(1.0, float(adaptive_max_batch_seconds))
        self._cache_enabled = bool(cache_enabled)
        self._cache_max_entries = max(1, int(cache_max_entries))
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._cache_evictions = 0
        self._torch_dtype = torch_dtype
        self._runtime_backend = str(runtime_backend or "hf").strip().lower()
        if self._runtime_backend not in {"hf", "vllm", "vllm_server"}:
            raise ValueError(f"Unsupported Chandra runtime_backend: {runtime_backend}")

        self._processor = None
        self._model = None
        self._vllm_llm = None
        self._vllm_constructor_kwargs: Dict[str, Any] = {}
        self._vllm_server_base_url = self._normalize_vllm_server_base_url(vllm_server_base_url)
        self._vllm_server_base_urls = self._normalize_vllm_server_base_urls(
            vllm_server_base_urls,
            default_url=self._vllm_server_base_url,
        )
        self._vllm_server_model = str(vllm_server_model or model_path)
        self._vllm_server_api_key = str(vllm_server_api_key or "").strip()
        self._vllm_server_timeout_seconds = max(1.0, float(vllm_server_timeout_seconds))
        self._vllm_server_max_retries = max(0, int(vllm_server_max_retries))
        self._vllm_server_retry_backoff_seconds = max(0.05, float(vllm_server_retry_backoff_seconds))
        self._vllm_server_parallel_requests = max(1, int(vllm_server_parallel_requests))
        self._vllm_server_probe_on_init = bool(vllm_server_probe_on_init)
        self._vllm_server_fallback_to_hf_on_error = bool(vllm_server_fallback_to_hf_on_error)
        self._vllm_server_circuit_breaker_threshold = max(1, int(vllm_server_circuit_breaker_threshold))
        self._vllm_server_consecutive_failures = 0
        self._vllm_server_circuit_open = False
        self._vllm_server_rr_index = 0
        self._vllm_server_rr_lock = threading.Lock()
        self._fallback_to_hf_count = 0

        if any(
            int(value) != self._fixed_max_new_tokens
            for value in (
                max_new_tokens,
                profile_speed_max_new_tokens,
                profile_balanced_max_new_tokens,
                profile_quality_max_new_tokens,
                adaptive_high_max_new_tokens,
            )
        ):
            logger.warning(
                "Chandra max_new_tokens runtime knobs are deprecated and ignored; using fixed output token budget=%s.",
                self._fixed_max_new_tokens,
            )

        if self._runtime_backend == "vllm":
            fallback_to_hf = bool(vllm_fallback_to_hf_on_error)
            if int(vllm_data_parallel_size) > 1:
                raise ValueError(
                    "ChandraDeployment runtime_backend='vllm' currently runs in-process via vllm.LLM. "
                    "vLLM data_parallel_size>1 is not supported in this mode. "
                    "Use tensor_parallel_size>1 or an external vLLM serve deployment for data parallelism."
                )
            try:
                from vllm import LLM  # type: ignore

                candidate_kwargs: Dict[str, Any] = {
                    "model": model_path,
                    "trust_remote_code": bool(vllm_trust_remote_code),
                    "tensor_parallel_size": int(vllm_tensor_parallel_size),
                    "data_parallel_size": int(vllm_data_parallel_size),
                    "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
                    "max_model_len": int(vllm_max_model_len),
                    "max_num_seqs": int(vllm_max_num_seqs),
                    "dtype": vllm_dtype,
                    "enforce_eager": bool(vllm_enforce_eager),
                    "disable_log_stats": bool(vllm_disable_log_stats),
                    "mm_encoder_tp_mode": vllm_mm_encoder_tp_mode,
                    "async_scheduling": bool(vllm_async_scheduling),
                    "enable_chunked_prefill": vllm_enable_chunked_prefill,
                }
                llm_signature = inspect.signature(LLM.__init__)
                accepted = set(llm_signature.parameters.keys())
                accepts_var_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in llm_signature.parameters.values()
                )
                self._vllm_constructor_kwargs = self._filter_vllm_constructor_kwargs(
                    candidate_kwargs=candidate_kwargs,
                    accepted_keys=accepted,
                    accepts_var_kwargs=accepts_var_kwargs,
                )
                self._vllm_llm = LLM(**self._vllm_constructor_kwargs)
                logger.info(
                    "Initialized Chandra vLLM runtime for model=%s with kwargs=%s",
                    model_path,
                    json.dumps(self._vllm_constructor_kwargs, sort_keys=True),
                )
            except Exception:
                if fallback_to_hf:
                    logger.exception(
                        "Chandra vLLM runtime initialization failed for model=%s. Falling back to HF runtime.",
                        model_path,
                    )
                    self._runtime_backend = "hf"
                else:
                    raise
        elif self._runtime_backend == "vllm_server":
            if self._vllm_server_probe_on_init:
                try:
                    self._probe_vllm_server_health()
                    logger.info(
                        "Initialized Chandra vLLM-server runtime at %s for model=%s",
                        ",".join(self._vllm_server_base_urls),
                        self._vllm_server_model,
                    )
                except Exception:
                    if self._vllm_server_fallback_to_hf_on_error:
                        logger.exception(
                            "Chandra vLLM-server initialization probe failed for endpoints=%s. Falling back to HF runtime.",
                            ",".join(self._vllm_server_base_urls),
                        )
                        self._runtime_backend = "hf"
                    else:
                        raise

        if self._runtime_backend == "hf":
            self._init_hf_runtime(
                model_path=model_path,
                max_image_pixels=max_image_pixels,
                torch_dtype=torch_dtype,
            )

        self._batcher = _MicroBatcher(self._process_batch, self._max_batch_size, self._max_batch_wait_ms)

    @staticmethod
    def _normalize_vllm_server_base_url(raw_url: str) -> str:
        value = str(raw_url or "").strip()
        if not value:
            raise ValueError("vllm_server_base_url must be provided for runtime_backend='vllm_server'.")
        if value.endswith("/"):
            value = value[:-1]
        if not value.endswith("/v1"):
            value = f"{value}/v1"
        return value

    @classmethod
    def _normalize_vllm_server_base_urls(
        cls,
        raw_urls: Optional[Any],
        default_url: str,
    ) -> List[str]:
        values: List[str] = []
        if isinstance(raw_urls, str):
            values = [entry.strip() for entry in raw_urls.split(",") if entry.strip()]
        elif isinstance(raw_urls, (list, tuple)):
            values = [str(entry).strip() for entry in raw_urls if str(entry).strip()]
        normalized: List[str] = []
        seen = set()
        for value in values:
            item = cls._normalize_vllm_server_base_url(value)
            if item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        if not normalized:
            return [cls._normalize_vllm_server_base_url(default_url)]
        return normalized

    def _next_vllm_server_base_url(self) -> str:
        endpoints = self._vllm_server_base_urls
        if len(endpoints) == 1:
            return endpoints[0]
        with self._vllm_server_rr_lock:
            selected = endpoints[self._vllm_server_rr_index % len(endpoints)]
            self._vllm_server_rr_index += 1
        return selected

    def _init_hf_runtime(
        self,
        model_path: str,
        max_image_pixels: Optional[int],
        torch_dtype: Optional[str],
    ) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            self._processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        if max_image_pixels:
            try:
                self._processor.image_processor.size = {
                    "shortest_edge": 65536,
                    "longest_edge": int(max_image_pixels),
                }
            except Exception:
                logger.warning("Unable to apply max_image_pixels to processor.")

        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype
        ).eval()
        try:
            self._model.to("cuda")
        except Exception:
            pass

    def _ensure_hf_runtime_initialized(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        self._init_hf_runtime(
            model_path=self._model_path,
            max_image_pixels=self._max_image_pixels,
            torch_dtype=self._torch_dtype,
        )

    def _probe_vllm_server_health(self) -> None:
        failures: List[str] = []
        for base_url in self._vllm_server_base_urls:
            health_url = f"{base_url.rsplit('/v1', 1)[0]}/health"
            request = urlrequest.Request(health_url, method="GET")
            try:
                with urlrequest.urlopen(request, timeout=min(self._vllm_server_timeout_seconds, 15.0)) as response:
                    if int(getattr(response, "status", 200)) >= 300:
                        raise RuntimeError(
                            f"vLLM server health probe failed with HTTP {getattr(response, 'status', 'unknown')}."
                        )
            except Exception as exc:
                failures.append(f"{health_url}: {exc}")
        if failures:
            raise RuntimeError("vLLM server health probe failures: " + " | ".join(failures))

    async def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        max_image_pixels: Optional[int] = None,
        profile: Optional[str] = None,
        adaptive_rerun_limit: Optional[int] = None,
        cache_bypass: bool = False,
    ) -> str:
        profile_name = str(profile or self._default_profile).strip().lower()
        if profile_name not in {"speed", "balanced", "quality", "adaptive"}:
            profile_name = self._default_profile if self._default_profile in {"speed", "balanced", "quality", "adaptive"} else "balanced"

        if profile_name == "adaptive":
            base_settings = self._profile_settings["speed"]
        else:
            base_settings = self._profile_settings.get(profile_name, self._profile_settings["balanced"])

        request_max_new_tokens = int(self._fixed_max_new_tokens)
        request_max_pixels = int(max_image_pixels if max_image_pixels is not None else base_settings["max_image_pixels"])
        if request_max_pixels <= 0:
            request_max_pixels = None

        allow_escalation = (
            profile_name == "adaptive"
            and max_image_pixels is None
            and self._adaptive_max_reruns_per_batch > 0
        )
        escalation_max_new_tokens = int(self._fixed_max_new_tokens)
        escalation_max_image_pixels = max(
            request_max_pixels or 0,
            self._adaptive_high_max_image_pixels,
        )
        if request_max_pixels is None:
            escalation_max_image_pixels = self._adaptive_high_max_image_pixels

        request_prompt = prompt or self._prompt
        request_cache_key = None
        if not cache_bypass:
            request_cache_key = self._build_cache_key(
                image=image,
                prompt=request_prompt,
                max_new_tokens=request_max_new_tokens,
                max_image_pixels=request_max_pixels,
                profile=profile_name,
                allow_escalation=allow_escalation,
                escalation_max_new_tokens=escalation_max_new_tokens,
                escalation_max_image_pixels=escalation_max_image_pixels,
            )
        if self._cache_enabled and request_cache_key is not None:
            cached = await self._cache_get(request_cache_key)
            if cached is not None:
                return cached

        request = ChandraRequest(
            image=image,
            prompt=request_prompt,
            max_new_tokens=request_max_new_tokens,
            max_image_pixels=request_max_pixels,
            cache_key=request_cache_key,
            allow_escalation=allow_escalation,
            escalation_max_new_tokens=escalation_max_new_tokens,
            escalation_max_image_pixels=escalation_max_image_pixels,
            adaptive_rerun_limit=max(0, int(adaptive_rerun_limit)) if adaptive_rerun_limit is not None else None,
        )
        return await self._batcher.submit(request)

    async def _process_batch(self, requests: List[ChandraRequest]) -> List[str]:
        batch_start = time.perf_counter()
        request_count = len(requests)
        results: List[Optional[str]] = [None] * request_count
        bucket_timings: List[dict] = []

        unique_indices: List[int] = []
        duplicate_map: Dict[int, List[int]] = {}
        seen_cache_keys: Dict[str, int] = {}
        for idx, req in enumerate(requests):
            if req.cache_key:
                first_idx = seen_cache_keys.get(req.cache_key)
                if first_idx is not None:
                    duplicate_map.setdefault(first_idx, []).append(idx)
                    continue
                seen_cache_keys[req.cache_key] = idx
            unique_indices.append(idx)

        uncached_unique_indices: List[int] = []
        for idx in unique_indices:
            cache_key = requests[idx].cache_key
            if self._cache_enabled and cache_key:
                cached = await self._cache_get(cache_key)
                if cached is not None:
                    results[idx] = cached
                    continue
            uncached_unique_indices.append(idx)
        cache_hit_count = request_count - len(uncached_unique_indices)
        cache_miss_count = len(uncached_unique_indices)

        loaded_images: Dict[int, Any] = {}
        prompts_by_idx: Dict[int, str] = {}
        load_start = time.perf_counter()
        for idx in uncached_unique_indices:
            req = requests[idx]
            loaded_images[idx] = self._load_image(req.image, max_image_pixels=req.max_image_pixels)
            prompts_by_idx[idx] = req.prompt
        load_seconds = time.perf_counter() - load_start

        first_pass_outputs: Dict[int, str] = {}
        if uncached_unique_indices:
            first_pass_outputs, first_timings = self._run_grouped_inference(
                indices=uncached_unique_indices,
                requests={idx: requests[idx] for idx in uncached_unique_indices},
                images_by_index=loaded_images,
                prompts_by_index=prompts_by_idx,
                stage_name="primary",
            )
            bucket_timings.extend(first_timings)
            for idx, value in first_pass_outputs.items():
                results[idx] = value

        escalated_indices: List[int] = []
        escalation_reason_counts: Dict[str, int] = {}
        escalation_dropped_budget_count = 0
        escalation_dropped_timeout_count = 0
        batch_rerun_budget_remaining = self._adaptive_max_reruns_per_batch
        if self._adaptive_max_reruns_per_batch > 0:
            for idx in uncached_unique_indices:
                req = requests[idx]
                output = results[idx]
                if not req.allow_escalation or output is None:
                    continue
                reasons = self._escalation_reasons(output)
                if not reasons:
                    continue
                for reason in reasons:
                    escalation_reason_counts[reason] = escalation_reason_counts.get(reason, 0) + 1
                rerun_cap_for_request = 1 if req.adaptive_rerun_limit is None else max(0, int(req.adaptive_rerun_limit))
                if rerun_cap_for_request <= 0:
                    escalation_dropped_budget_count += 1
                    continue
                if batch_rerun_budget_remaining <= 0:
                    escalation_dropped_budget_count += 1
                    continue
                if (time.perf_counter() - batch_start) >= self._adaptive_max_batch_seconds:
                    escalation_dropped_timeout_count += 1
                    continue
                escalated_indices.append(idx)
                batch_rerun_budget_remaining -= 1

        if escalated_indices:
            if (time.perf_counter() - batch_start) >= self._adaptive_max_batch_seconds:
                escalation_dropped_timeout_count += len(escalated_indices)
                escalated_indices = []
        if escalated_indices:
            escalation_images: Dict[int, Any] = {}
            escalation_prompts: Dict[int, str] = {}
            escalation_requests: List[ChandraRequest] = []
            escalation_index_map: List[int] = []
            for idx in escalated_indices:
                req = requests[idx]
                escalation_max_image_pixels = req.escalation_max_image_pixels or req.max_image_pixels
                escalation_max_new_tokens = req.escalation_max_new_tokens or req.max_new_tokens
                escalation_req = ChandraRequest(
                    image=req.image,
                    prompt=req.prompt,
                    max_new_tokens=int(escalation_max_new_tokens),
                    max_image_pixels=int(escalation_max_image_pixels) if escalation_max_image_pixels else None,
                )
                escalation_requests.append(escalation_req)
                escalation_index_map.append(idx)

            escalation_load_start = time.perf_counter()
            for local_idx, req in enumerate(escalation_requests):
                request_idx = escalation_index_map[local_idx]
                escalation_images[request_idx] = self._load_image(
                    req.image,
                    max_image_pixels=req.max_image_pixels,
                )
                escalation_prompts[request_idx] = req.prompt
            escalation_load_seconds = time.perf_counter() - escalation_load_start
            load_seconds += escalation_load_seconds

            if (time.perf_counter() - batch_start) >= self._adaptive_max_batch_seconds:
                escalation_dropped_timeout_count += len(escalation_index_map)
            else:
                escalation_output_map, escalation_timings = self._run_grouped_inference(
                    indices=escalation_index_map,
                    requests={
                        idx: req for idx, req in zip(escalation_index_map, escalation_requests)
                    },
                    images_by_index=escalation_images,
                    prompts_by_index=escalation_prompts,
                    stage_name="escalation",
                )
                bucket_timings.extend(escalation_timings)
                for idx, value in escalation_output_map.items():
                    results[idx] = value

        for first_idx, duplicates in duplicate_map.items():
            first_value = results[first_idx]
            for duplicate_idx in duplicates:
                results[duplicate_idx] = first_value

        missing_results = [idx for idx, value in enumerate(results) if value is None]
        if missing_results:
            raise RuntimeError(f"Missing Chandra outputs for indices: {missing_results}")

        if self._cache_enabled:
            for idx, value in enumerate(results):
                req = requests[idx]
                if req.cache_key and value is not None:
                    await self._cache_set(req.cache_key, value)

        total_seconds = time.perf_counter() - batch_start
        if self._enable_timing_logs:
            all_images_for_stats = [
                loaded_images.get(idx, None)
                for idx in uncached_unique_indices
                if idx in loaded_images
            ]
            self._log_batch_timing(
                requests=requests,
                images=all_images_for_stats,
                load_seconds=load_seconds,
                total_seconds=total_seconds,
                bucket_timings=bucket_timings,
                escalated_count=len(escalated_indices),
                cache_hit_count=cache_hit_count,
                cache_miss_count=cache_miss_count,
                duplicate_count=sum(len(v) for v in duplicate_map.values()),
                escalation_reason_counts=escalation_reason_counts,
                escalation_dropped_budget_count=escalation_dropped_budget_count,
                escalation_dropped_timeout_count=escalation_dropped_timeout_count,
            )
        return [value for value in results if value is not None]

    def _load_image(self, image: Any, max_image_pixels: Optional[int] = None):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to load images") from exc

        if image is None:
            raise ValueError("Image is required")
        if isinstance(image, Image.Image):
            return self._cap_image_pixels(image, max_image_pixels=max_image_pixels)
        if isinstance(image, (bytes, bytearray)):
            return self._cap_image_pixels(Image.open(io.BytesIO(image)).convert("RGB"), max_image_pixels=max_image_pixels)
        if isinstance(image, str):
            if image.strip().startswith("data:image"):
                b64 = image.split(",", 1)[-1]
                return self._cap_image_pixels(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), max_image_pixels=max_image_pixels)
            return self._cap_image_pixels(Image.open(image).convert("RGB"), max_image_pixels=max_image_pixels)
        raise ValueError("Unsupported image type for Chandra")

    def _cap_image_pixels(self, image, max_image_pixels: Optional[int] = None):
        cap = self._max_image_pixels if max_image_pixels is None else int(max_image_pixels)
        if not cap:
            return image
        try:
            from PIL import Image
        except Exception:  # pragma: no cover
            return image
        width, height = image.size
        pixels = width * height
        if pixels <= cap:
            return image
        scale = (cap / float(pixels)) ** 0.5
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _build_size_buckets(self, images: List[Any]) -> List[List[int]]:
        indexed_areas = []
        for idx, image in enumerate(images):
            width, height = image.size
            indexed_areas.append((idx, max(1, width * height)))
        indexed_areas.sort(key=lambda pair: (pair[1], pair[0]))

        buckets: List[List[int]] = []
        current_bucket: List[int] = []
        current_min_area = 1

        for idx, area in indexed_areas:
            if not current_bucket:
                current_bucket = [idx]
                current_min_area = area
                continue

            would_overflow = len(current_bucket) >= self._max_batch_size
            size_mismatch = area > (current_min_area * self._size_bucket_ratio)
            if would_overflow or size_mismatch:
                buckets.append(current_bucket)
                current_bucket = [idx]
                current_min_area = area
                continue
            current_bucket.append(idx)

        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    def _run_grouped_inference(
        self,
        indices: List[int],
        requests: Dict[int, ChandraRequest],
        images_by_index: Dict[int, Any],
        prompts_by_index: Dict[int, str],
        stage_name: str,
    ) -> Tuple[Dict[int, str], List[dict]]:
        outputs: Dict[int, str] = {}
        timings: List[dict] = []
        token_groups: Dict[int, List[int]] = {}
        for idx in sorted(indices):
            token_groups.setdefault(int(requests[idx].max_new_tokens), []).append(idx)

        for token_limit, group_indices in sorted(token_groups.items(), key=lambda entry: entry[0]):
            group_images = [images_by_index[idx] for idx in group_indices]
            group_prompts = [prompts_by_index[idx] for idx in group_indices]
            group_buckets = self._build_size_buckets(group_images)
            for bucket_id, local_bucket_indices in enumerate(group_buckets):
                request_indices = [group_indices[local_idx] for local_idx in local_bucket_indices]
                bucket_images = [images_by_index[idx] for idx in request_indices]
                bucket_prompts = [prompts_by_index[idx] for idx in request_indices]
                decoded, timing = self._run_bucket_inference(
                    images=bucket_images,
                    prompts=bucket_prompts,
                    max_new_tokens=token_limit,
                    stage_name=stage_name,
                    bucket_id=bucket_id,
                )
                for request_idx, text in zip(request_indices, decoded):
                    outputs[request_idx] = text
                timing["token_limit"] = int(token_limit)
                timings.append(timing)
        return outputs, timings

    def _run_bucket_inference(
        self,
        images: List[Any],
        prompts: List[str],
        max_new_tokens: int,
        stage_name: str,
        bucket_id: int,
    ) -> Tuple[List[str], dict]:
        if self._runtime_backend == "vllm":
            return self._run_bucket_inference_vllm(
                images=images,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                stage_name=stage_name,
                bucket_id=bucket_id,
            )
        if self._runtime_backend == "vllm_server":
            return self._run_bucket_inference_vllm_server(
                images=images,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                stage_name=stage_name,
                bucket_id=bucket_id,
            )
        return self._run_bucket_inference_hf(
            images=images,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            stage_name=stage_name,
            bucket_id=bucket_id,
        )

    def _run_bucket_inference_hf(
        self,
        images: List[Any],
        prompts: List[str],
        max_new_tokens: int,
        stage_name: str,
        bucket_id: int,
    ) -> Tuple[List[str], dict]:
        bucket_start = time.perf_counter()
        chat_start = time.perf_counter()
        chat_texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_texts.append(
                self._processor.apply_chat_template(messages, add_generation_prompt=True)  # type: ignore[union-attr]
            )
        chat_seconds = time.perf_counter() - chat_start

        encode_start = time.perf_counter()
        inputs = self._processor(  # type: ignore[operator]
            text=chat_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        encode_seconds = time.perf_counter() - encode_start

        move_start = time.perf_counter()
        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]
        except Exception:
            pass
        move_seconds = time.perf_counter() - move_start

        generate_start = time.perf_counter()
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[union-attr]
        generate_seconds = time.perf_counter() - generate_start

        decode_start = time.perf_counter()
        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)  # type: ignore[union-attr]
        decode_seconds = time.perf_counter() - decode_start

        clean_start = time.perf_counter()
        cleaned = [self._clean_output(text, prompt) for text, prompt in zip(decoded, prompts)]
        clean_seconds = time.perf_counter() - clean_start

        bucket_areas = [image.size[0] * image.size[1] for image in images]
        timing = {
            "stage": stage_name,
            "bucket_id": int(bucket_id),
            "size": len(images),
            "pixel_min": min(bucket_areas) if bucket_areas else 0,
            "pixel_max": max(bucket_areas) if bucket_areas else 0,
            "chat_ms": round(chat_seconds * 1000.0, 2),
            "encode_ms": round(encode_seconds * 1000.0, 2),
            "move_ms": round(move_seconds * 1000.0, 2),
            "generate_ms": round(generate_seconds * 1000.0, 2),
            "decode_ms": round(decode_seconds * 1000.0, 2),
            "clean_ms": round(clean_seconds * 1000.0, 2),
            "total_ms": round((time.perf_counter() - bucket_start) * 1000.0, 2),
        }
        return cleaned, timing

    def _run_bucket_inference_vllm(
        self,
        images: List[Any],
        prompts: List[str],
        max_new_tokens: int,
        stage_name: str,
        bucket_id: int,
    ) -> Tuple[List[str], dict]:
        if self._vllm_llm is None:
            raise RuntimeError("vLLM backend requested, but vLLM runtime is not initialized.")

        from vllm import SamplingParams  # type: ignore

        bucket_start = time.perf_counter()
        payload_start = time.perf_counter()
        batch_messages: List[List[Dict[str, Any]]] = []
        for image, prompt in zip(images, prompts):
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_pil", "image_pil": image},
                        ],
                    }
                ]
            )
        payload_seconds = time.perf_counter() - payload_start

        sampling_params = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=0.0,
            top_p=1.0,
        )
        infer_start = time.perf_counter()
        decoded_raw: List[str] = []
        try:
            outputs = self._vllm_llm.chat(messages=batch_messages, sampling_params=sampling_params)
            decoded_raw = [self._extract_vllm_output_text(entry) for entry in outputs]
            if len(decoded_raw) != len(batch_messages):
                raise RuntimeError(
                    f"vLLM batch output length mismatch ({len(decoded_raw)} != {len(batch_messages)})."
                )
        except Exception:
            logger.exception(
                "vLLM batched chat call failed for bucket=%s stage=%s; retrying sequentially.",
                bucket_id,
                stage_name,
            )
            decoded_raw = []
            for single_message in batch_messages:
                output = self._vllm_llm.chat(messages=single_message, sampling_params=sampling_params)
                if not output:
                    decoded_raw.append("")
                    continue
                decoded_raw.append(self._extract_vllm_output_text(output[0]))
        infer_seconds = time.perf_counter() - infer_start

        clean_start = time.perf_counter()
        cleaned = [self._clean_output(text, prompt) for text, prompt in zip(decoded_raw, prompts)]
        clean_seconds = time.perf_counter() - clean_start

        bucket_areas = [image.size[0] * image.size[1] for image in images]
        timing = {
            "stage": stage_name,
            "bucket_id": int(bucket_id),
            "size": len(images),
            "pixel_min": min(bucket_areas) if bucket_areas else 0,
            "pixel_max": max(bucket_areas) if bucket_areas else 0,
            "chat_ms": round(payload_seconds * 1000.0, 2),
            "encode_ms": 0.0,
            "move_ms": 0.0,
            "generate_ms": round(infer_seconds * 1000.0, 2),
            "decode_ms": 0.0,
            "clean_ms": round(clean_seconds * 1000.0, 2),
            "total_ms": round((time.perf_counter() - bucket_start) * 1000.0, 2),
        }
        return cleaned, timing

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _extract_openai_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type in {"text", "output_text"} and item.get("text") is not None:
                        text_parts.append(str(item.get("text")))
                    elif item.get("content") is not None:
                        text_parts.append(str(item.get("content")))
                elif item is not None:
                    text_parts.append(str(item))
            return "\n".join(part for part in text_parts if part).strip()
        if content is None:
            return ""
        return str(content)

    def _request_vllm_server_completion(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        selected_base_url = self._next_vllm_server_base_url()
        endpoint = f"{selected_base_url}/chat/completions"
        payload = {
            "model": self._vllm_server_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                    ],
                }
            ],
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._vllm_server_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_server_api_key}"

        last_exception: Optional[Exception] = None
        for attempt in range(self._vllm_server_max_retries + 1):
            request = urlrequest.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urlrequest.urlopen(request, timeout=self._vllm_server_timeout_seconds) as response:
                    status_code = int(getattr(response, "status", 200))
                    response_body = response.read().decode("utf-8")
                if status_code >= 300:
                    raise RuntimeError(f"vLLM server returned HTTP {status_code}: {response_body[:512]}")

                parsed = json.loads(response_body)
                choices = parsed.get("choices") or []
                if not choices:
                    raise RuntimeError("vLLM server response missing choices.")
                first_choice = choices[0] or {}
                message = first_choice.get("message") if isinstance(first_choice, dict) else None
                if message is None:
                    # Fallback for older response shape.
                    text_value = first_choice.get("text", "") if isinstance(first_choice, dict) else ""
                    return str(text_value)
                content = message.get("content") if isinstance(message, dict) else message
                return self._extract_openai_message_content(content)
            except urlerror.HTTPError as exc:
                last_exception = exc
            except Exception as exc:
                last_exception = exc

            if attempt < self._vllm_server_max_retries:
                sleep_seconds = self._vllm_server_retry_backoff_seconds * (2 ** attempt)
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"vLLM server request failed at {selected_base_url} "
            f"after {self._vllm_server_max_retries + 1} attempts: {last_exception}"
        )

    def _run_bucket_inference_vllm_server(
        self,
        images: List[Any],
        prompts: List[str],
        max_new_tokens: int,
        stage_name: str,
        bucket_id: int,
    ) -> Tuple[List[str], dict]:
        if len(images) != len(prompts):
            raise ValueError("vLLM server inference received mismatched images/prompts lengths.")

        if self._vllm_server_circuit_open:
            if self._vllm_server_fallback_to_hf_on_error:
                logger.warning(
                    "Chandra vLLM-server circuit is open; using HF fallback for bucket=%s stage=%s.",
                    bucket_id,
                    stage_name,
                )
                self._fallback_to_hf_count += 1
                self._ensure_hf_runtime_initialized()
                return self._run_bucket_inference_hf(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    stage_name=stage_name,
                    bucket_id=bucket_id,
                )
            raise RuntimeError("Chandra vLLM-server circuit is open and fallback is disabled.")

        bucket_start = time.perf_counter()
        infer_start = time.perf_counter()
        decoded_raw: List[str] = []
        try:
            max_workers = min(len(images), self._vllm_server_parallel_requests)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._request_vllm_server_completion,
                        image,
                        prompt,
                        int(max_new_tokens),
                    )
                    for image, prompt in zip(images, prompts)
                ]
                decoded_raw = [future.result() for future in futures]
            self._vllm_server_consecutive_failures = 0
        except Exception as exc:
            self._vllm_server_consecutive_failures += 1
            if self._vllm_server_consecutive_failures >= self._vllm_server_circuit_breaker_threshold:
                self._vllm_server_circuit_open = True
            logger.exception(
                "vLLM-server inference failed (bucket=%s stage=%s, failures=%s): %s",
                bucket_id,
                stage_name,
                self._vllm_server_consecutive_failures,
                exc,
            )
            if self._vllm_server_fallback_to_hf_on_error:
                self._fallback_to_hf_count += 1
                self._ensure_hf_runtime_initialized()
                return self._run_bucket_inference_hf(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    stage_name=stage_name,
                    bucket_id=bucket_id,
                )
            raise
        infer_seconds = time.perf_counter() - infer_start

        clean_start = time.perf_counter()
        cleaned = [self._clean_output(text, prompt) for text, prompt in zip(decoded_raw, prompts)]
        clean_seconds = time.perf_counter() - clean_start

        bucket_areas = [image.size[0] * image.size[1] for image in images]
        timing = {
            "stage": stage_name,
            "bucket_id": int(bucket_id),
            "size": len(images),
            "pixel_min": min(bucket_areas) if bucket_areas else 0,
            "pixel_max": max(bucket_areas) if bucket_areas else 0,
            "chat_ms": 0.0,
            "encode_ms": 0.0,
            "move_ms": 0.0,
            "generate_ms": round(infer_seconds * 1000.0, 2),
            "decode_ms": 0.0,
            "clean_ms": round(clean_seconds * 1000.0, 2),
            "total_ms": round((time.perf_counter() - bucket_start) * 1000.0, 2),
            "server_retries": int(self._vllm_server_max_retries),
        }
        return cleaned, timing

    @staticmethod
    def _extract_vllm_output_text(output: Any) -> str:
        if output is None:
            return ""
        if hasattr(output, "outputs") and output.outputs:
            first = output.outputs[0]
            return str(getattr(first, "text", ""))
        return str(output)

    @staticmethod
    def _filter_vllm_constructor_kwargs(
        candidate_kwargs: Dict[str, Any], accepted_keys: set, accepts_var_kwargs: bool = False
    ) -> Dict[str, Any]:
        filtered: Dict[str, Any] = {}
        for key, value in candidate_kwargs.items():
            if not accepts_var_kwargs and key not in accepted_keys:
                continue
            if value is None:
                continue
            filtered[key] = value
        return filtered

    def _escalation_reasons(self, text: str) -> List[str]:
        stripped = (text or "").strip()
        reasons: List[str] = []
        if len(stripped) < self._adaptive_min_chars:
            reasons.append("short_output")
        if stripped.count("```") % 2 == 1:
            reasons.append("unclosed_fence")
        if stripped.count("<table") > stripped.count("</table>"):
            reasons.append("html_table_unclosed")
        if stripped.count("<div") > stripped.count("</div>"):
            reasons.append("html_div_unclosed")
        if stripped.count("\\begin{") > stripped.count("\\end{"):
            reasons.append("math_unclosed")
        suspect_tails = {
            "<table": "html_tail_table",
            "<tr": "html_tail_tr",
            "<td": "html_tail_td",
            "<th": "html_tail_th",
            "<div": "html_tail_div",
            "<span": "html_tail_span",
            "\\begin{": "math_tail",
            "| --- |": "table_tail_header",
        }
        for token, reason in suspect_tails.items():
            if stripped.endswith(token):
                reasons.append(reason)
                break
        return reasons

    async def _cache_get(self, key: str) -> Optional[str]:
        async with self._cache_lock:
            value = self._cache.get(key)
            if value is None:
                return None
            self._cache.move_to_end(key)
            return value

    async def _cache_set(self, key: str, value: str) -> None:
        async with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)
                self._cache_evictions += 1

    def _build_cache_key(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int,
        max_image_pixels: Optional[int],
        profile: str,
        allow_escalation: bool,
        escalation_max_new_tokens: Optional[int],
        escalation_max_image_pixels: Optional[int],
    ) -> Optional[str]:
        if not self._cache_enabled:
            return None
        image_digest = self._digest_image_input(image)
        if image_digest is None:
            return None
        payload = {
            "model": self._model_path,
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "max_image_pixels": int(max_image_pixels) if max_image_pixels else None,
            "profile": profile,
            "allow_escalation": bool(allow_escalation),
            "escalation_max_new_tokens": int(escalation_max_new_tokens) if escalation_max_new_tokens else None,
            "escalation_max_image_pixels": int(escalation_max_image_pixels) if escalation_max_image_pixels else None,
            "image_sha256": image_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _digest_image_input(self, image: Any) -> Optional[str]:
        try:
            from PIL import Image
        except Exception:  # pragma: no cover
            return None

        try:
            if image is None:
                return None
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return hashlib.sha256(buffer.getvalue()).hexdigest()
            if isinstance(image, (bytes, bytearray)):
                return hashlib.sha256(bytes(image)).hexdigest()
            if isinstance(image, str):
                if image.strip().startswith("data:image"):
                    b64 = image.split(",", 1)[-1]
                    return hashlib.sha256(base64.b64decode(b64)).hexdigest()
                if os.path.exists(image):
                    with open(image, "rb") as handle:
                        return hashlib.sha256(handle.read()).hexdigest()
                return hashlib.sha256(image.encode("utf-8")).hexdigest()
            return hashlib.sha256(repr(image).encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _log_batch_timing(
        self,
        requests: List[ChandraRequest],
        images: List[Any],
        load_seconds: float,
        total_seconds: float,
        bucket_timings: List[dict],
        escalated_count: int = 0,
        cache_hit_count: int = 0,
        cache_miss_count: int = 0,
        duplicate_count: int = 0,
        escalation_reason_counts: Optional[Dict[str, int]] = None,
        escalation_dropped_budget_count: int = 0,
        escalation_dropped_timeout_count: int = 0,
    ) -> None:
        image_pixels = [image.size[0] * image.size[1] for image in images]
        max_tokens = max((req.max_new_tokens for req in requests), default=0)
        payload = {
            "event": "chandra_batch_timing",
            "runtime_backend": self._runtime_backend,
            "batch_size": len(requests),
            "max_new_tokens": int(max_tokens),
            "load_ms": round(load_seconds * 1000.0, 2),
            "total_ms": round(total_seconds * 1000.0, 2),
            "pixel_min": min(image_pixels) if image_pixels else 0,
            "pixel_max": max(image_pixels) if image_pixels else 0,
            "pixel_mean": round(sum(image_pixels) / len(image_pixels), 2) if image_pixels else 0,
            "bucket_count": len(bucket_timings),
            "escalated_count": int(escalated_count),
            "cache_hit_count": int(cache_hit_count),
            "cache_miss_count": int(cache_miss_count),
            "duplicate_count": int(duplicate_count),
            "cache_current_size": len(self._cache),
            "cache_evictions": int(self._cache_evictions),
            "fallback_to_hf_count": int(self._fallback_to_hf_count),
            "vllm_server_circuit_open": bool(self._vllm_server_circuit_open),
            "vllm_server_consecutive_failures": int(self._vllm_server_consecutive_failures),
            "vllm_server_endpoints": list(self._vllm_server_base_urls),
            "escalation_reason_counts": dict(escalation_reason_counts or {}),
            "escalation_dropped_budget_count": int(escalation_dropped_budget_count),
            "escalation_dropped_timeout_count": int(escalation_dropped_timeout_count),
            "buckets": bucket_timings,
        }
        logger.info("CHANDRA_TIMING %s", json.dumps(payload, sort_keys=True))

    def _clean_output(self, text: str, prompt: str) -> str:
        if not text:
            return text
        cleaned = text.strip()
        lines = cleaned.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].strip().lower() == "user":
            lines.pop(0)
            if lines and lines[0].strip() == prompt:
                lines.pop(0)
            if lines and lines[0].strip().lower() == "assistant":
                lines.pop(0)
        elif lines and lines[0].strip().lower() == "assistant":
            lines.pop(0)
        lines = self._strip_prompt_echo(lines, prompt)
        while lines and not lines[0].strip():
            lines.pop(0)
        cleaned = "\n".join(lines).strip()
        cleaned = self._unescape_markdown(cleaned)
        cleaned = self._normalize_tables(cleaned)
        if re.search(
            r"</?(?:div|p|br|table|tr|td|th|math|sup|sub|img|ul|ol|li|h[1-6]|pre|code)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            try:
                from markdownify import markdownify as _markdownify
            except Exception:
                return cleaned
            try:
                return self._unescape_markdown(_markdownify(cleaned))
            except Exception:
                return cleaned
        return cleaned

    def _unescape_markdown(self, text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        out: List[str] = []
        in_fence = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                out.append(line)
                continue
            if in_fence or self._line_has_math(line):
                out.append(line)
                continue
            parts = line.split("`")
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    continue
                parts[idx] = re.sub(r"\\([\\`*_{}\[\]()#+.!|<>:=/@?,\\-])", r"\1", part)
            fixed = "`".join(parts)
            if fixed.rstrip().endswith("</"):
                fixed = fixed.rstrip()[:-2].rstrip()
            out.append(fixed)
        return "\n".join(out)

    @staticmethod
    def _line_has_math(line: str) -> bool:
        if "$" in line:
            return True
        tokens = ("\\\\(", "\\\\)", "\\\\[", "\\\\]", "\\\\begin{", "\\\\end{")
        return any(tok in line for tok in tokens)

    @staticmethod
    def _strip_prompt_echo(lines: List[str], prompt: str) -> List[str]:
        if not lines or not prompt:
            return lines
        prompt_lines = [ln for ln in prompt.splitlines() if ln.strip()]
        if not prompt_lines:
            return lines

        def normalize(line: str) -> str:
            normalized = line.strip()
            if normalized.startswith("\\"):
                normalized = normalized.lstrip("\\").lstrip()
            if normalized.startswith(("-", "*", "•")):
                normalized = normalized[1:].lstrip()
            return normalized

        idx = 0
        pidx = 0
        while idx < len(lines) and pidx < len(prompt_lines):
            if not lines[idx].strip():
                idx += 1
                continue
            if normalize(lines[idx]) == normalize(prompt_lines[pidx]):
                idx += 1
                pidx += 1
                continue
            break
        if pidx == len(prompt_lines):
            lines = lines[idx:]
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines and lines[0].strip().lower() == "assistant":
                lines.pop(0)
                while lines and not lines[0].strip():
                    lines.pop(0)
        return lines

    @staticmethod
    def _normalize_tables(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        out: List[str] = []
        i = 0
        separator_re = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")
        while i < len(lines):
            line = lines[i]
            if "|" in line and i + 1 < len(lines) and separator_re.match(lines[i + 1]):
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                rows = [row.strip().strip("|").split("|") for row in table_lines]
                max_cols = max(len(row) for row in rows) if rows else 0
                keep_cols = [False] * max_cols
                for row in rows:
                    for idx, cell in enumerate(row):
                        if cell.strip():
                            keep_cols[idx] = True
                if any(keep_cols):
                    normalized = []
                    for row in rows:
                        row = row + [""] * (max_cols - len(row))
                        kept = [cell.strip() for idx, cell in enumerate(row) if keep_cols[idx]]
                        normalized.append("| " + " | ".join(kept) + " |")
                    out.extend(normalized)
                else:
                    out.extend(table_lines)
                continue
            out.append(line)
            i += 1
        return "\n".join(out)
