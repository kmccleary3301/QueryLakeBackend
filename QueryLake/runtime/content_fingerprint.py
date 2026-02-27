from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_fingerprint(*, text: str, md: Dict[str, Any] | None = None, salt: str = "") -> str:
    payload = {
        "text": text or "",
        "md": md or {},
        "salt": salt or "",
    }
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
