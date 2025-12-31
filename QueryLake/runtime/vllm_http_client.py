from __future__ import annotations

from typing import AsyncGenerator, Dict

import httpx
from fastapi.responses import JSONResponse, StreamingResponse


def _prepare_headers(raw_headers: Dict[str, str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    auth = raw_headers.get("authorization") or raw_headers.get("Authorization")
    if auth:
        headers["authorization"] = auth
    return headers


async def proxy_vllm_chat_completion(
    base_url: str,
    request_body: Dict,
    raw_headers: Dict[str, str],
):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = _prepare_headers(raw_headers)

    if request_body.get("stream"):
        async def _stream() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=request_body, headers=headers) as resp:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=request_body, headers=headers)
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        return JSONResponse(content={"raw": resp.text}, status_code=resp.status_code)


async def proxy_vllm_embeddings(
    base_url: str,
    request_body: Dict,
    raw_headers: Dict[str, str],
):
    url = base_url.rstrip("/") + "/v1/embeddings"
    headers = _prepare_headers(raw_headers)

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=request_body, headers=headers)
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        return JSONResponse(content={"raw": resp.text}, status_code=resp.status_code)

