#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProbeResult:
    status: str
    stage: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def _safe_import_torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        return str(torch.__version__)
    except Exception:
        return None


def _load_probe_image(pdf_path: Path, page_number: int):
    from PIL import Image  # type: ignore
    import pypdfium2 as pdfium  # type: ignore

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_idx = max(0, page_number - 1)
    if page_idx >= len(pdf):
        raise ValueError(f"Requested page {page_number}, but PDF has {len(pdf)} pages.")
    page = pdf[page_idx]
    image = page.render(scale=2.0).to_pil().convert("RGB")
    page.close()
    pdf.close()
    return image


def _run_generate_probe(
    llm: Any,
    image: Any,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    attempts = []
    outputs_text = None

    if hasattr(llm, "chat"):
        chat_payloads = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_pil", "image_pil": image},
                    ],
                }
            ],
        ]
        for messages in chat_payloads:
            start = time.perf_counter()
            try:
                outputs = llm.chat(messages=messages, sampling_params=params)
                elapsed = time.perf_counter() - start
                text = ""
                if outputs and hasattr(outputs[0], "outputs") and outputs[0].outputs:
                    text = outputs[0].outputs[0].text
                elif outputs:
                    text = str(outputs[0])
                attempts.append(
                    {
                        "status": "ok",
                        "api": "chat",
                        "elapsed_seconds": round(elapsed, 4),
                        "text_chars": len(text),
                    }
                )
                outputs_text = text
                break
            except Exception as exc:
                elapsed = time.perf_counter() - start
                attempts.append(
                    {
                        "status": "error",
                        "api": "chat",
                        "elapsed_seconds": round(elapsed, 4),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        if outputs_text is not None:
            return {"attempts": attempts, "text": outputs_text}

    prompt_variants = [
        ("raw", prompt),
        ("lt_image", f"<image>\n{prompt}"),
        ("vision_tokens", f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt}"),
        ("image_1", f"<|image_1|>\n{prompt}"),
    ]
    for prompt_name, prompt_text in prompt_variants:
        # Try common multimodal payload shapes across vLLM versions.
        payload_attempts = [
            [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
            {"prompt": prompt_text, "multi_modal_data": {"image": image}},
            [{"prompt": prompt_text, "multi_modal_data": {"image": [image]}}],
        ]
        for payload in payload_attempts:
            start = time.perf_counter()
            try:
                outputs = llm.generate(payload, sampling_params=params)
                elapsed = time.perf_counter() - start
                text = ""
                if outputs and hasattr(outputs[0], "outputs") and outputs[0].outputs:
                    text = outputs[0].outputs[0].text
                elif outputs:
                    text = str(outputs[0])
                attempts.append(
                    {
                        "status": "ok",
                        "prompt_variant": prompt_name,
                        "payload_type": type(payload).__name__,
                        "elapsed_seconds": round(elapsed, 4),
                        "text_chars": len(text),
                    }
                )
                outputs_text = text
                break
            except Exception as exc:  # pragma: no cover
                elapsed = time.perf_counter() - start
                attempts.append(
                    {
                        "status": "error",
                        "prompt_variant": prompt_name,
                        "payload_type": type(payload).__name__,
                        "elapsed_seconds": round(elapsed, 4),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        if outputs_text is not None:
            break
    return {"attempts": attempts, "text": outputs_text}


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "unknown",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "conda_env": os.getenv("CONDA_DEFAULT_ENV"),
            "torch_version": _safe_import_torch_version(),
        },
        "input": {
            "model": args.model,
            "pdf": str(args.pdf),
            "page": args.page,
            "max_tokens": args.max_tokens,
            "trust_remote_code": args.trust_remote_code,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
        },
        "stages": {},
    }

    try:
        import vllm  # type: ignore
        from vllm import LLM  # type: ignore
    except Exception as exc:
        failure = ProbeResult(
            status="failed",
            stage="import",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
        report["status"] = "failed"
        report["stages"]["import"] = asdict(failure)
        report["timing_seconds"] = round(time.perf_counter() - started, 4)
        return report

    report["environment"]["vllm_version"] = getattr(vllm, "__version__", "unknown")

    load_start = time.perf_counter()
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
        )
        report["stages"]["load"] = asdict(
            ProbeResult(
                status="ok",
                stage="load",
                details={"elapsed_seconds": round(time.perf_counter() - load_start, 4)},
            )
        )
    except Exception as exc:
        report["status"] = "failed"
        report["stages"]["load"] = asdict(
            ProbeResult(
                status="failed",
                stage="load",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
                details={"elapsed_seconds": round(time.perf_counter() - load_start, 4)},
            )
        )
        report["timing_seconds"] = round(time.perf_counter() - started, 4)
        return report

    image_start = time.perf_counter()
    try:
        image = _load_probe_image(Path(args.pdf), args.page)
        report["stages"]["render_probe_page"] = asdict(
            ProbeResult(
                status="ok",
                stage="render_probe_page",
                details={
                    "elapsed_seconds": round(time.perf_counter() - image_start, 4),
                    "image_size": getattr(image, "size", None),
                },
            )
        )
    except Exception as exc:
        report["status"] = "failed"
        report["stages"]["render_probe_page"] = asdict(
            ProbeResult(
                status="failed",
                stage="render_probe_page",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        report["timing_seconds"] = round(time.perf_counter() - started, 4)
        return report

    gen_start = time.perf_counter()
    try:
        gen_result = _run_generate_probe(
            llm=llm,
            image=image,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        ok = gen_result.get("text") is not None
        report["stages"]["generate"] = asdict(
            ProbeResult(
                status="ok" if ok else "failed",
                stage="generate",
                details={
                    "elapsed_seconds": round(time.perf_counter() - gen_start, 4),
                    "attempts": gen_result.get("attempts", []),
                    "output_text_chars": len(gen_result.get("text") or ""),
                },
            )
        )
        report["status"] = "ok" if ok else "failed"
    except Exception as exc:  # pragma: no cover
        report["status"] = "failed"
        report["stages"]["generate"] = asdict(
            ProbeResult(
                status="failed",
                stage="generate",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
                details={"elapsed_seconds": round(time.perf_counter() - gen_start, 4)},
            )
        )

    report["timing_seconds"] = round(time.perf_counter() - started, 4)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Chandra compatibility on a specific vLLM environment."
    )
    parser.add_argument("--model", required=True, help="Model path/id.")
    parser.add_argument("--pdf", required=True, help="Local PDF path for one-page smoke generation.")
    parser.add_argument("--page", type=int, default=1, help="1-based page number to probe.")
    parser.add_argument("--prompt", default="Convert this page into clean Markdown.", help="Probe prompt.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--worker-timeout-seconds", type=int, default=240)
    parser.add_argument("--out-json", default=None, help="Optional output JSON path.")
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-report", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_mode:
        report = run_probe(args)
        output = json.dumps(report, indent=2)
        if args.worker_report:
            report_path = Path(args.worker_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(output, encoding="utf-8")
        print(output)
        if report.get("status") != "ok":
            raise SystemExit(1)
        return

    with tempfile.NamedTemporaryFile(prefix="chandra_vllm_probe_", suffix=".json", delete=False) as tmp:
        worker_report = tmp.name

    cmd = [
        sys.executable,
        __file__,
        "--worker-mode",
        "--worker-report",
        worker_report,
        "--model",
        args.model,
        "--pdf",
        str(args.pdf),
        "--page",
        str(args.page),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--dtype",
        str(args.dtype),
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(args.worker_timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
        )
    wrapper_seconds = round(time.perf_counter() - started, 4)

    report: Dict[str, Any]
    report_path = Path(worker_report)
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {
                "status": "failed",
                "stage": "wrapper_parse_worker_report",
                "error_type": "JSONDecodeError",
                "error_message": "Worker report existed but could not be parsed.",
            }
    else:
        report = {
            "status": "failed",
            "stage": "wrapper_no_worker_report",
            "error_type": "WorkerTerminated",
            "error_message": "Worker exited before writing structured report.",
        }

    report["wrapper"] = {
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": wrapper_seconds,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "command": cmd,
    }

    output = json.dumps(report, indent=2)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    print(output)

    try:
        report_path.unlink(missing_ok=True)
    except Exception:
        pass

    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
