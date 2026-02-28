from __future__ import annotations

from typing import Any, Optional


class QueryLakeError(Exception):
    """Base exception for all SDK failures."""


class QueryLakeTransportError(QueryLakeError):
    """Raised when a request cannot be completed due to transport issues."""


class QueryLakeHTTPStatusError(QueryLakeError):
    """Raised when QueryLake returns a non-2xx status code."""

    def __init__(self, status_code: int, url: str, body: str):
        self.status_code = int(status_code)
        self.url = url
        self.body = body
        super().__init__(f"HTTP {self.status_code} from {self.url}: {self.body}")


class QueryLakeAPIError(QueryLakeError):
    """Raised when QueryLake returns {'success': false} from /api routes."""

    def __init__(
        self,
        *,
        function_name: str,
        message: str,
        trace: Optional[str] = None,
        payload: Optional[Any] = None,
    ):
        self.function_name = function_name
        self.message = message
        self.trace = trace
        self.payload = payload
        super().__init__(f"API '{self.function_name}' failed: {self.message}")
