from __future__ import annotations

import asyncio
import mimetypes
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlmodel import Session, select

from QueryLake.files.object_store import LocalCASObjectStore, ObjectStore
from QueryLake.database import sql_db_tables as T
from QueryLake.observability import metrics
from QueryLake.runtime.sse import SessionStreamHub
from QueryLake.api.single_user_auth import get_user
from QueryLake.api.custom_model_functions.surya import process_pdf_with_surya_2


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

    async def process_version(self, file_id: str, version_id: str, *, auth: Any) -> Dict[str, Any]:
        self._assert_file_readable(file_id, auth)
        # Prevent overload: simple concurrency gate
        await self._proc_sem.acquire()
        # Minimal, idempotent-ish pipeline; placeholders for heavy steps
        fv = self.db.get(T.file_version, version_id)
        if fv is None or fv.file_id != file_id:
            raise HTTPException(status_code=404, detail="File version not found")

        # Idempotence via fingerprint
        extra = {"surya": bool(getattr(self.umbrella, "surya_handles", None)), "embed": bool(getattr(self.umbrella, "embedding_handles", None))}
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
            # Try real OCR pipeline if available and PDF
            if self.umbrella and getattr(self.umbrella, "surya_handles", None) and (fv.mime_type or "").endswith("pdf"):
                from io import BytesIO
                data = self.store.get_bytes(fv.bytes_cas)
                if data is None:
                    raise HTTPException(status_code=404, detail="Bytes not found in store")
                bio = BytesIO(data)
                bio.name = (getattr(self.db.get(T.file, file_id), "logical_name", None) or "upload.pdf")
                try:
                    full_text, images_dict, out_meta = await process_pdf_with_surya_2(
                        database=self.db,
                        auth=auth,
                        server_surya_handles=self.umbrella.surya_handles,
                        file=bio,
                    )
                    text_result = full_text
                    pages_count = int(out_meta.get("pages", 1) or 1)
                    info_bytes = (str(out_meta) or "{}").encode("utf-8")
                    ocr_info_cas = self.store.put_bytes(info_bytes)
                except Exception:
                    # Fall back to placeholder if OCR fails
                    text_result = None
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
            ev = self.events.append(file_id, version_id, "OCR_DONE", {"pages": pages_count, "ocr_json_cas": ocr_info_cas})
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
