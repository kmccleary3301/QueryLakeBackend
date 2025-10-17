from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


class ObjectStore:
    """Abstract object store with CAS primitives."""

    def put_bytes(self, data: bytes) -> str:
        raise NotImplementedError

    def get_bytes(self, cas: str) -> Optional[bytes]:
        raise NotImplementedError

    def exists(self, cas: str) -> bool:
        raise NotImplementedError


class LocalCASObjectStore(ObjectStore):
    """Simple filesystem-backed CAS store.

    Layout: <base_dir>/<sha256[:2]>/<sha256>
    """

    def __init__(self, base_dir: str | Path = "object_store") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _path_for(self, cas: str) -> Path:
        shard = cas[:2]
        return self.base_dir / shard / cas

    def put_bytes(self, data: bytes) -> str:
        cas = self._sha256(data)
        path = self._path_for(cas)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return cas

    def get_bytes(self, cas: str) -> Optional[bytes]:
        path = self._path_for(cas)
        if not path.exists():
            return None
        return path.read_bytes()

    def exists(self, cas: str) -> bool:
        return self._path_for(cas).exists()

