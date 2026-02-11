from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlmodel import Session, select

from QueryLake.files.object_store import LocalCASObjectStore, ObjectStore
from QueryLake.database import sql_db_tables as T
from QueryLake.observability import metrics
from QueryLake.runtime.sse import SessionStreamHub
from QueryLake.api.single_user_auth import get_user
try:
    from QueryLake.api.custom_model_functions.surya import process_pdf_with_surya_2
except Exception:  # pragma: no cover
    process_pdf_with_surya_2 = None


FILE_EVENT_KINDS = (
    "FILE_UPLOADED",
    "SAFETY_SCANNED",
    "OCR_DONE",
    "TEXT_NORMALIZED",
    "CHUNKED",
    "EMBEDDED",
    "INDEXED",
    "FAILED",
)


@dataclass
class FileEvent:
    file_id: str
    version_id: str
    rev: int
    kind: str
    payload: Dict[str, Any]
    ts: float


class FileEventStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _next_rev(self, file_id: str) -> int:
        result = self.db.exec(
            select(T.file_event.rev).where(T.file_event.file_id == file_id).order_by(T.file_event.rev.desc()).limit(1)
        ).first()
        return 1 if result is None else (int(result) + 1)

    def append(self, file_id: str, version_id: str, kind: str, payload: Dict[str, Any]) -> FileEvent:
        rev = self._next_rev(file_id)
        row = T.file_event(
            file_id=file_id,
            version_id=version_id,
            rev=rev,
            ts=time.time(),
            kind=kind,
            payload=payload,
        )
        self.db.add(row)
        self.db.commit()
        metrics.inc_event(kind)
        return FileEvent(file_id=file_id, version_id=version_id, rev=rev, kind=kind, payload=payload, ts=row.ts)

    def list(self, file_id: str, since: Optional[int] = None) -> List[FileEvent]:
        stmt = select(T.file_event).where(T.file_event.file_id == file_id).order_by(T.file_event.rev.asc())
        if since is not None:
            stmt = stmt.where(T.file_event.rev > since)
        records = self.db.exec(stmt).all()
        out: List[FileEvent] = []
        for r in records:
            out.append(FileEvent(file_id=r.file_id, version_id=r.version_id, rev=r.rev, kind=r.kind, payload=r.payload, ts=r.ts))
        return out

    def upsert_job(self, job_id: str, file_id: str, version_id: str, status: str, *, progress: Optional[dict] = None, result_meta: Optional[dict] = None) -> None:
        existing = self.db.get(T.file_job, job_id)
        if existing is None:
            row = T.file_job(
                job_id=job_id,
                file_id=file_id,
                version_id=version_id,
                status=status,
                progress=progress,
                result_meta=result_meta,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self.db.add(row)
        else:
            existing.status = status
            existing.progress = progress if progress is not None else existing.progress
            existing.result_meta = result_meta if result_meta is not None else existing.result_meta
            existing.updated_at = time.time()
        self.db.commit()
        metrics.job_transition(status, "files")

    def list_jobs(self, file_id: str) -> List[Dict[str, Any]]:
        stmt = select(T.file_job).where(T.file_job.file_id == file_id).order_by(T.file_job.updated_at.desc())
        records = self.db.exec(stmt).all()
        out: List[Dict[str, Any]] = []
        for r in records:
            out.append(
                {
                    "job_id": r.job_id,
                    "file_id": r.file_id,
                    "version_id": r.version_id,
                    "status": r.status,
                    "progress": r.progress,
                    "result_meta": r.result_meta,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                }
            )
        return out

    def dead_letter(self, file_id: str, version_id: Optional[str], event: Dict[str, Any], error: str) -> None:
        row = T.file_dead_letter(file_id=file_id, version_id=version_id, event=event, error=error)
        self.db.add(row)
        self.db.commit()


class FilesRuntimeService:
    """Phase 1–2 Files runtime.

    - Phase 1: CAS store, SQL metadata, dual-write compatibility
    - Phase 2: Event-sourced processing + simple jobs
    """

    def __init__(self, database: Session, object_store: Optional[ObjectStore] = None, umbrella: Any = None) -> None:
        self.db = database
        self.store = object_store or LocalCASObjectStore()
        self.events = FileEventStore(self.db)
        self.sse = SessionStreamHub()  # Keyed by file_id
        self.umbrella = umbrella
        self._proc_sem = asyncio.Semaphore(2)
        self._render_cache: "OrderedDict[str, str]" = OrderedDict()
        self._render_cache_lock = asyncio.Lock()
        self._render_cache_max_entries = max(
            1,
            int(os.getenv("QUERYLAKE_CHANDRA_RENDER_CACHE_MAX_ENTRIES", "512") or 512),
        )
        self._default_chandra_profile = os.getenv("QUERYLAKE_CHANDRA_PROFILE", "balanced").strip() or "balanced"
        self._default_chandra_dpi = max(
            72,
            int(os.getenv("QUERYLAKE_CHANDRA_RENDER_DPI", "144") or 144),
        )
        self._default_chandra_concurrency = max(
            1,
            int(os.getenv("QUERYLAKE_CHANDRA_CONCURRENCY", "24") or 24),
        )
        # Default is "off" because the text layer is typically lower-quality than Chandra OCR
        # (layout loss, missing figures, etc.). Enable explicitly when you want it.
        text_layer_mode = (os.getenv("QUERYLAKE_PDF_TEXT_LAYER_MODE", "off") or "off").strip().lower()
        if text_layer_mode not in {"off", "auto", "prefer", "mixed"}:
            text_layer_mode = "auto"
        self._pdf_text_layer_mode = text_layer_mode
        self._pdf_text_min_chars_per_page = max(
            1,
            int(os.getenv("QUERYLAKE_PDF_TEXT_MIN_CHARS_PER_PAGE", "120") or 120),
        )
        self._pdf_text_min_coverage = float(
            os.getenv("QUERYLAKE_PDF_TEXT_MIN_COVERAGE", "0.80") or 0.80
        )
        self._pdf_text_min_coverage = min(1.0, max(0.0, self._pdf_text_min_coverage))

    @staticmethod
    def _compute_render_cache_key(bytes_cas: str, dpi: int, page_num: int) -> str:
        payload = f"{bytes_cas}|dpi={dpi}|page={page_num}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def _render_cache_get(self, key: str) -> Optional[str]:
        async with self._render_cache_lock:
            cas = self._render_cache.get(key)
            if cas is None:
                return None
            self._render_cache.move_to_end(key)
            return cas

    async def _render_cache_set(self, key: str, cas: str) -> None:
        async with self._render_cache_lock:
            self._render_cache[key] = cas
            self._render_cache.move_to_end(key)
            while len(self._render_cache) > self._render_cache_max_entries:
                self._render_cache.popitem(last=False)

    async def _select_chandra_handle(self) -> Optional[Any]:
        handles = getattr(self.umbrella, "chandra_handles", None) if self.umbrella else None
        if not handles:
            return None
        preferred_id = os.getenv("QUERYLAKE_DEFAULT_CHANDRA_ID", "").strip()
        if preferred_id and preferred_id in handles:
            return handles[preferred_id]
        # Prefer id-like keys over name aliases where possible.
        for key, handle in handles.items():
            if key.lower() == key and " " not in key:
                return handle
        return next(iter(handles.values()))

    @staticmethod
    def _normalize_pdf_text_layer_page(text: Optional[str]) -> str:
        if not text:
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join([line.rstrip() for line in normalized.split("\n")]).strip()

    def _evaluate_pdf_text_layer_candidate(self, page_texts: List[str]) -> Dict[str, Any]:
        page_count = len(page_texts)
        page_chars = [len((text or "").strip()) for text in page_texts]
        non_empty_pages = sum(1 for value in page_chars if value > 0)
        qualified_pages = sum(1 for value in page_chars if value >= self._pdf_text_min_chars_per_page)
        total_chars = int(sum(page_chars))
        avg_chars_per_page = (float(total_chars) / float(page_count)) if page_count > 0 else 0.0
        coverage = (float(qualified_pages) / float(page_count)) if page_count > 0 else 0.0
        mode = self._pdf_text_layer_mode

        selected = False
        reason = "disabled"
        if mode == "off":
            selected = False
            reason = "mode_off"
        elif mode == "prefer":
            selected = non_empty_pages > 0
            reason = "prefer_nonempty" if selected else "prefer_but_empty"
        else:
            # Auto mode selects text layer for clearly digital PDFs.
            selected = (
                coverage >= self._pdf_text_min_coverage
                and avg_chars_per_page >= float(self._pdf_text_min_chars_per_page)
            )
            reason = "auto_threshold_met" if selected else "auto_threshold_miss"

        return {
            "mode": mode,
            "selected": selected,
            "reason": reason,
            "pages": page_count,
            "total_chars": total_chars,
            "avg_chars_per_page": round(avg_chars_per_page, 2),
            "qualified_page_coverage": round(coverage, 4),
            "qualified_pages": qualified_pages,
            "non_empty_pages": non_empty_pages,
            "min_chars_per_page": self._pdf_text_min_chars_per_page,
            "min_coverage": self._pdf_text_min_coverage,
        }

    def _evaluate_pdf_text_layer_page_overrides(self, page_texts: List[str]) -> Dict[str, Any]:
        mode = self._pdf_text_layer_mode
        page_count = len(page_texts)
        page_chars = [len((text or "").strip()) for text in page_texts]
        if mode == "off":
            return {
                "mode": mode,
                "selected": False,
                "reason": "mode_off",
                "selected_page_indices": [],
                "selected_pages": 0,
                "selected_coverage": 0.0,
                "page_count": page_count,
                "min_chars_for_page_select": self._pdf_text_min_chars_per_page,
            }

        if mode == "prefer":
            threshold = 1
            reason = "prefer_nonempty_pages"
        elif mode == "mixed":
            threshold = self._pdf_text_min_chars_per_page
            reason = "mixed_min_chars_threshold"
        else:
            return {
                "mode": mode,
                "selected": False,
                "reason": "mode_not_page_routed",
                "selected_page_indices": [],
                "selected_pages": 0,
                "selected_coverage": 0.0,
                "page_count": page_count,
                "min_chars_for_page_select": self._pdf_text_min_chars_per_page,
            }

        selected_indices = [idx for idx, value in enumerate(page_chars) if value >= threshold]
        selected_count = len(selected_indices)
        coverage = (float(selected_count) / float(page_count)) if page_count > 0 else 0.0
        return {
            "mode": mode,
            "selected": selected_count > 0,
            "reason": reason if selected_count > 0 else f"{reason}_none",
            "selected_page_indices": selected_indices,
            "selected_pages": selected_count,
            "selected_coverage": round(coverage, 4),
            "page_count": page_count,
            "min_chars_for_page_select": threshold,
        }

    def _extract_pdf_text_layer_pages(self, data: bytes) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"engine": "pdf_text_layer", "status": "not_selected"}
        if self._pdf_text_layer_mode == "off":
            meta.update({"reason": "mode_off", "mode": self._pdf_text_layer_mode})
            return None, meta

        try:
            from io import BytesIO
            from pypdf import PdfReader
        except Exception as exc:
            meta.update(
                {
                    "reason": "dependency_unavailable",
                    "error": str(exc),
                    "mode": self._pdf_text_layer_mode,
                }
            )
            return None, meta

        try:
            reader = PdfReader(BytesIO(data))
            page_texts = [
                self._normalize_pdf_text_layer_page(page.extract_text())
                for page in reader.pages
            ]
        except Exception as exc:
            meta.update(
                {
                    "reason": "extract_error",
                    "error": str(exc),
                    "mode": self._pdf_text_layer_mode,
                }
            )
            return None, meta

        meta.update(
            {
                "mode": self._pdf_text_layer_mode,
                "pages": len(page_texts),
                "status": "parsed",
            }
        )
        return page_texts, meta

    def _try_extract_pdf_text_layer(self, data: bytes) -> Tuple[Optional[str], Dict[str, Any]]:
        page_texts, meta = self._extract_pdf_text_layer_pages(data)
        if page_texts is None:
            return None, meta

        decision = self._evaluate_pdf_text_layer_candidate(page_texts)
        meta.update(decision)
        if not decision.get("selected"):
            return None, meta

        page_blocks: List[str] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            body = page_text if page_text else ""
            page_blocks.append(f"## Page {page_idx}\n\n{body}".strip())
        return "\n\n".join(page_blocks), meta

    async def _process_pdf_with_chandra(
        self,
        data: bytes,
        bytes_cas: str,
        *,
        profile: str,
        dpi: int,
        concurrency: int,
        page_text_overrides: Optional[Dict[int, str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from io import BytesIO
        from PIL import Image
        import pypdfium2 as pdfium

        page_text_overrides = page_text_overrides or {}

        pdf_bytes = BytesIO(data)
        pdf = pdfium.PdfDocument(pdf_bytes)
        page_count = len(pdf)
        scale = float(dpi) / 72.0

        ocr_jobs: List[Tuple[int, Any]] = []
        render_hits = 0
        render_misses = 0
        outputs: List[str] = [""] * page_count

        for page_idx, text_override in page_text_overrides.items():
            if 0 <= int(page_idx) < page_count:
                outputs[int(page_idx)] = str(text_override or "")

        for page_idx in range(page_count):
            if page_idx in page_text_overrides:
                continue
            page_num = page_idx + 1
            cache_key = self._compute_render_cache_key(bytes_cas, dpi, page_num)
            cached_cas = await self._render_cache_get(cache_key)
            if cached_cas is not None:
                image_bytes = self.store.get_bytes(cached_cas)
                if image_bytes is not None:
                    render_hits += 1
                    ocr_jobs.append((page_idx, Image.open(BytesIO(image_bytes)).convert("RGB")))
                    continue
            page = pdf[page_idx]
            image = page.render(scale=scale).to_pil().convert("RGB")
            page.close()
            render_misses += 1
            target = BytesIO()
            image.save(target, format="PNG")
            image_cas = self.store.put_bytes(target.getvalue())
            await self._render_cache_set(cache_key, image_cas)
            ocr_jobs.append((page_idx, image))
        pdf.close()

        if ocr_jobs:
            handle = await self._select_chandra_handle()
            if handle is None:
                raise RuntimeError("No Chandra handle available.")

            semaphore = asyncio.Semaphore(max(1, int(concurrency)))

            async def _run_page(page_index: int, image_obj: Any) -> None:
                async with semaphore:
                    outputs[page_index] = await handle.transcribe.remote(
                        image=image_obj,
                        profile=profile,
                    )

            await asyncio.gather(*[_run_page(idx, img) for idx, img in ocr_jobs])

        full_text = "\n\n".join(outputs)
        return full_text, {
            "engine": "chandra",
            "profile": profile,
            "pages": page_count,
            "ocr_pages": len(ocr_jobs),
            "text_layer_pages": len(page_text_overrides),
            "render_cache_hits": render_hits,
            "render_cache_misses": render_misses,
            "render_cache_size": len(self._render_cache),
            "render_dpi": dpi,
        }

    # ------------------
    # Access control
    # ------------------
    def _username_from_auth(self, auth: Any) -> Optional[str]:
        try:
            _, ua = get_user(self.db, auth)
            return ua.username
        except Exception:
            # If not resolvable, deny by default
            return None

    def _assert_file_readable(self, file_id: str, auth: Any) -> None:
        username = self._username_from_auth(auth)
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        f = self.db.get(T.file, file_id)
        if f is None:
            raise HTTPException(status_code=404, detail="File not found")
        if f.created_by and f.created_by != username:
            # TODO: extend to collection/org membership
            raise HTTPException(status_code=403, detail="Not authorized for file")

    async def upload_file(
        self,
        auth: Any,
        file: UploadFile,
        *,
        logical_name: Optional[str] = None,
        collection_id: Optional[str] = None,
        checksum_sha256: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Read bytes
        data = await file.read()
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload")

        # Optional checksum verify
        if checksum_sha256:
            import hashlib
            calc = hashlib.sha256(data).hexdigest()
            if calc != checksum_sha256:
                raise HTTPException(status_code=400, detail="Checksum mismatch")

        # Content-addressed store
        bytes_cas = self.store.put_bytes(data)
        size_bytes = len(data)
        mime_type = file.content_type or (mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream")

        # Idempotent dedup: if an identical CAS already exists for this user, return it
        username = self._username_from_auth(auth)
        if username:
            from sqlmodel import select
            candidates = self.db.exec(select(T.file_version).where(T.file_version.bytes_cas == bytes_cas)).all()
            for cand in candidates:
                parent = self.db.get(T.file, cand.file_id)
                if parent and parent.created_by == username:
                    return {
                        "file_id": parent.id,
                        "version_id": cand.id,
                        "version_no": cand.version_no,
                        "bytes_cas": cand.bytes_cas,
                        "size_bytes": cand.size_bytes,
                        "mime_type": cand.mime_type,
                        "note": "deduplicated",
                    }

        # Create or reuse file row
        logical_name = logical_name or (file.filename or "unnamed")
        f = T.file(
            logical_name=logical_name,
            created_at=time.time(),
            created_by=getattr(auth, "username", None),
            collection_id=collection_id,
        )
        self.db.add(f)
        self.db.commit()
        self.db.refresh(f)

        # Determine next version number
        stmt = select(T.file_version.version_no).where(T.file_version.file_id == f.id).order_by(T.file_version.version_no.desc()).limit(1)
        last = self.db.exec(stmt).first()
        version_no = 1 if last is None else int(last) + 1

        fv = T.file_version(
            file_id=f.id,
            version_no=version_no,
            bytes_cas=bytes_cas,
            size_bytes=size_bytes,
            mime_type=mime_type,
            created_at=time.time(),
        )
        self.db.add(fv)
        self.db.commit()
        self.db.refresh(fv)

        ev = self.events.append(f.id, fv.id, "FILE_UPLOADED", {"bytes_cas": bytes_cas, "size": size_bytes, "mime": mime_type})
        await self._publish(f.id, ev)

        return {
            "file_id": f.id,
            "version_id": fv.id,
            "version_no": version_no,
            "bytes_cas": bytes_cas,
            "size_bytes": size_bytes,
            "mime_type": mime_type,
        }

    async def list_file(self, file_id: str, *, auth: Any) -> Dict[str, Any]:
        self._assert_file_readable(file_id, auth)
        f = self.db.get(T.file, file_id)
        if f is None:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "id": f.id,
            "logical_name": f.logical_name,
            "collection_id": f.collection_id,
            "created_at": f.created_at,
            "created_by": f.created_by,
        }

    async def list_versions(self, file_id: str, *, auth: Any) -> List[Dict[str, Any]]:
        self._assert_file_readable(file_id, auth)
        stmt = select(T.file_version).where(T.file_version.file_id == file_id).order_by(T.file_version.version_no.asc())
        recs = self.db.exec(stmt).all()
        return [
            {
                "id": r.id,
                "file_id": r.file_id,
                "version_no": r.version_no,
                "bytes_cas": r.bytes_cas,
                "size_bytes": r.size_bytes,
                "mime_type": r.mime_type,
                "created_at": r.created_at,
            }
            for r in recs
        ]

    async def list_events(self, file_id: str, since: Optional[int], *, auth: Any) -> List[Dict[str, Any]]:
        self._assert_file_readable(file_id, auth)
        events = self.events.list(file_id, since)
        return [
            {"rev": e.rev, "kind": e.kind, "payload": e.payload, "ts": e.ts}
            for e in events
        ]

    async def list_jobs(self, file_id: str, *, auth: Any) -> List[Dict[str, Any]]:
        self._assert_file_readable(file_id, auth)
        return self.events.list_jobs(file_id)

    async def subscribe(self, file_id: str):
        return await self.sse.subscribe(file_id)

    async def unsubscribe(self, file_id: str, subscriber) -> None:
        await self.sse.unsubscribe(file_id, subscriber)

    async def _publish(self, file_id: str, event: FileEvent) -> None:
        await self.sse.publish(
            file_id,
            {
                "event": event.kind,
                "data": {"rev": event.rev, "kind": event.kind, "payload": event.payload, "ts": event.ts},
                "rev": event.rev,
            },
        )

    @staticmethod
    def compute_fingerprint(bytes_cas: str, pipeline: str = "files_v1", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import hashlib, json
        payload = {"pipeline": pipeline, "bytes_cas": bytes_cas, **(extra or {})}
        sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return {"fingerprint": payload, "sha": sha}

    async def process_version(
        self,
        file_id: str,
        version_id: str,
        *,
        auth: Any,
        ocr_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._assert_file_readable(file_id, auth)
        # Prevent overload: simple concurrency gate
        await self._proc_sem.acquire()
        # Minimal, idempotent-ish pipeline; placeholders for heavy steps
        fv = self.db.get(T.file_version, version_id)
        if fv is None or fv.file_id != file_id:
            raise HTTPException(status_code=404, detail="File version not found")

        # Idempotence via fingerprint
        selected_profile = (ocr_profile or self._default_chandra_profile).strip() or "balanced"
        extra = {
            "surya": bool(getattr(self.umbrella, "surya_handles", None)),
            "chandra": bool(getattr(self.umbrella, "chandra_handles", None)),
            "embed": bool(getattr(self.umbrella, "embedding_handles", None)),
            "ocr_profile": selected_profile,
            "chandra_dpi": self._default_chandra_dpi,
            "pdf_text_layer_mode": self._pdf_text_layer_mode,
            "pdf_text_layer_min_chars_per_page": self._pdf_text_min_chars_per_page,
            "pdf_text_layer_min_coverage": self._pdf_text_min_coverage,
        }
        fp = self.compute_fingerprint(fv.bytes_cas, extra=extra)
        if fv.processing_fingerprint == fp:
            # Already processed under same config; short-circuit
            return {"job_id": None, "status": "COMPLETED", "note": "already_processed"}

        job_id = f"jb_{int(time.time()*1000)}"
        self.events.upsert_job(job_id, file_id, version_id, "RUNNING")

        try:
            fv.processing_fingerprint = fp
            self.db.add(fv)
            self.db.commit()
            # Safety scan (placeholder)
            ev = self.events.append(file_id, version_id, "SAFETY_SCANNED", {"ok": True})
            await self._publish(file_id, ev)

            pages_count = 1
            text_result = None
            ocr_info_cas = None
            ocr_engine = "none"
            render_cache_hits = 0
            render_cache_misses = 0
            # Try Chandra first for PDFs when available; fall back to Surya.
            if self.umbrella and (fv.mime_type or "").endswith("pdf"):
                from io import BytesIO
                data = self.store.get_bytes(fv.bytes_cas)
                if data is None:
                    raise HTTPException(status_code=404, detail="Bytes not found in store")
                text_layer_text, text_layer_meta = self._try_extract_pdf_text_layer(data)
                if text_layer_text is not None:
                    text_result = text_layer_text
                    pages_count = int(text_layer_meta.get("pages", 1) or 1)
                    ocr_engine = "pdf_text_layer"
                    info_bytes = json.dumps(text_layer_meta, sort_keys=True).encode("utf-8")
                    ocr_info_cas = self.store.put_bytes(info_bytes)
                mixed_page_overrides: Optional[Dict[int, str]] = None
                mixed_routing_meta: Optional[Dict[str, Any]] = None
                if (
                    text_result is None
                    and self._pdf_text_layer_mode == "mixed"
                ):
                    page_texts, parsed_meta = self._extract_pdf_text_layer_pages(data)
                    if page_texts is not None:
                        page_routing = self._evaluate_pdf_text_layer_page_overrides(page_texts)
                        selected_indices = list(page_routing.get("selected_page_indices", []))
                        if selected_indices:
                            mixed_page_overrides = {
                                int(idx): page_texts[int(idx)]
                                for idx in selected_indices
                            }
                        mixed_routing_meta = {
                            **parsed_meta,
                            "routing": {
                                key: value
                                for key, value in page_routing.items()
                                if key != "selected_page_indices"
                            },
                            "selected_page_indices": selected_indices,
                        }
                chandra_failed = None
                if text_result is None and getattr(self.umbrella, "chandra_handles", None):
                    try:
                        full_text, out_meta = await self._process_pdf_with_chandra(
                            data,
                            fv.bytes_cas,
                            profile=selected_profile,
                            dpi=self._default_chandra_dpi,
                            concurrency=self._default_chandra_concurrency,
                            page_text_overrides=mixed_page_overrides,
                        )
                        text_result = full_text
                        pages_count = int(out_meta.get("pages", 1) or 1)
                        render_cache_hits = int(out_meta.get("render_cache_hits", 0) or 0)
                        render_cache_misses = int(out_meta.get("render_cache_misses", 0) or 0)
                        ocr_engine = "chandra_mixed" if mixed_page_overrides else "chandra"
                        if mixed_routing_meta is not None:
                            out_meta = {
                                **out_meta,
                                "text_layer_routing": mixed_routing_meta,
                            }
                        info_bytes = json.dumps(out_meta, sort_keys=True).encode("utf-8")
                        ocr_info_cas = self.store.put_bytes(info_bytes)
                    except Exception as exc:
                        chandra_failed = str(exc)

                if text_result is None and getattr(self.umbrella, "surya_handles", None):
                    bio = BytesIO(data)
                    bio.name = (getattr(self.db.get(T.file, file_id), "logical_name", None) or "upload.pdf")
                    try:
                        if process_pdf_with_surya_2 is None:
                            raise RuntimeError("Surya OCR code is not available in this environment.")
                        full_text, images_dict, out_meta = await process_pdf_with_surya_2(
                            database=self.db,
                            auth=auth,
                            server_surya_handles=self.umbrella.surya_handles,
                            file=bio,
                        )
                        text_result = full_text
                        pages_count = int(out_meta.get("pages", 1) or 1)
                        ocr_engine = "surya"
                        if chandra_failed:
                            out_meta = {**out_meta, "chandra_error": chandra_failed}
                        info_bytes = json.dumps(out_meta, sort_keys=True).encode("utf-8")
                        ocr_info_cas = self.store.put_bytes(info_bytes)
                    except Exception:
                        text_result = None
                        ocr_info_cas = self.store.put_bytes(b"{}")
                elif text_result is None:
                    ocr_info_cas = self.store.put_bytes(b"{}")
            else:
                ocr_info_cas = self.store.put_bytes(b"{}")

            # Persist a single page row for now (extend later to all pages)
            page = T.file_page(
                file_version_id=version_id,
                page_num=1,
                width_px=None,
                height_px=None,
                ocr_json_cas=ocr_info_cas,
                image_cas=None,
            )
            self.db.add(page)
            self.db.commit()
            ev = self.events.append(
                file_id,
                version_id,
                "OCR_DONE",
                {
                    "pages": pages_count,
                    "ocr_json_cas": ocr_info_cas,
                    "engine": ocr_engine,
                    "profile": selected_profile,
                    "render_cache_hits": render_cache_hits,
                    "render_cache_misses": render_cache_misses,
                },
            )
            await self._publish(file_id, ev)

            # Normalize text (placeholder)
            ev = self.events.append(file_id, version_id, "TEXT_NORMALIZED", {"notes": "placeholder"})
            await self._publish(file_id, ev)

            # Chunking (placeholder or single chunk from text_result)
            chunk = T.file_chunk(
                file_version_id=version_id,
                page_start=1,
                page_end=1,
                byte_start=0,
                byte_end=0,
                text=(text_result if text_result else f"CAS:{fv.bytes_cas} size:{fv.size_bytes}"),
                md={},
                anchors=[],
                embedding=None,
            )
            self.db.add(chunk)
            self.db.commit()
            ev = self.events.append(file_id, version_id, "CHUNKED", {"chunks": 1})
            await self._publish(file_id, ev)

            # Embedding / Indexing (placeholders)
            # Optional: compute embedding via umbrella if available
            embedded_count = 0
            try:
                if self.umbrella and getattr(self.umbrella, "embedding_handles", None):
                    # choose default embedding model
                    default_emb = getattr(self.umbrella.config.default_models, "embedding", None)
                    handle = self.umbrella.embedding_handles.get(default_emb) if default_emb else None
                    if handle is not None:
                        vecs = await handle.embed.remote([chunk.text])
                        # Assume vecs -> List[List[float]]
                        vec = vecs[0] if vecs else None
                        if vec:
                            chunk.embedding = vec
                            self.db.add(chunk)
                            self.db.commit()
                            embedded_count = 1
            except Exception:
                # Non-fatal in this phase
                pass

            ev = self.events.append(file_id, version_id, "EMBEDDED", {"dim": 1024, "count": embedded_count})
            await self._publish(file_id, ev)
            ev = self.events.append(file_id, version_id, "INDEXED", {"bm25": True, "hnsw": True})
            await self._publish(file_id, ev)

            self.events.upsert_job(job_id, file_id, version_id, "COMPLETED")
            return {"job_id": job_id, "status": "COMPLETED"}
        except Exception as e:
            self.events.upsert_job(job_id, file_id, version_id, "FAILED", result_meta={"error": str(e)})
            ev = self.events.append(file_id, version_id, "FAILED", {"error": str(e)})
            await self._publish(file_id, ev)
            try:
                self.events.dead_letter(file_id, version_id, {"stage": ev.kind, "payload": ev.payload}, str(e))
            except Exception:
                pass
            raise
        finally:
            self._proc_sem.release()
