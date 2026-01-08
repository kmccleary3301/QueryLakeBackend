from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from ray import serve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChandraRequest:
    image: Any
    prompt: str
    max_new_tokens: int


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
    """Serve deployment for Chandra with microbatching and optional image downscale cap."""

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
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 25,
        max_new_tokens: int = 2048,
        max_image_pixels: Optional[int] = 2_097_152,
        torch_dtype: Optional[str] = "auto",
    ) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self._model_path = model_path
        self._prompt = prompt
        self._max_new_tokens = int(max_new_tokens)
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_ms = int(max_batch_wait_ms)
        self._max_image_pixels = int(max_image_pixels) if max_image_pixels else None

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

        self._batcher = _MicroBatcher(self._process_batch, self._max_batch_size, self._max_batch_wait_ms)

    async def transcribe(self, image: Any, prompt: Optional[str] = None, max_new_tokens: Optional[int] = None) -> str:
        request = ChandraRequest(
            image=image,
            prompt=prompt or self._prompt,
            max_new_tokens=int(max_new_tokens or self._max_new_tokens),
        )
        return await self._batcher.submit(request)

    async def _process_batch(self, requests: List[ChandraRequest]) -> List[str]:
        images = [self._load_image(req.image) for req in requests]
        prompts = [req.prompt for req in requests]
        max_new_tokens = max(req.max_new_tokens for req in requests)
        # Qwen3-VL expects chat template with image placeholders.
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
                self._processor.apply_chat_template(messages, add_generation_prompt=True)
            )

        inputs = self._processor(
            text=chat_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
        return [self._clean_output(text, prompt) for text, prompt in zip(decoded, prompts)]

    def _load_image(self, image: Any):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to load images") from exc

        if image is None:
            raise ValueError("Image is required")
        if isinstance(image, Image.Image):
            return self._cap_image_pixels(image)
        if isinstance(image, (bytes, bytearray)):
            return self._cap_image_pixels(Image.open(io.BytesIO(image)).convert("RGB"))
        if isinstance(image, str):
            if image.strip().startswith("data:image"):
                b64 = image.split(",", 1)[-1]
                return self._cap_image_pixels(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
            return self._cap_image_pixels(Image.open(image).convert("RGB"))
        raise ValueError("Unsupported image type for Chandra")

    def _cap_image_pixels(self, image):
        if not self._max_image_pixels:
            return image
        try:
            from PIL import Image
        except Exception:  # pragma: no cover
            return image
        width, height = image.size
        pixels = width * height
        if pixels <= self._max_image_pixels:
            return image
        scale = (self._max_image_pixels / float(pixels)) ** 0.5
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
