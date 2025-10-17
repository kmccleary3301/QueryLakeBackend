#!/usr/bin/env python3
"""Simple helper to hit the OpenAI-compatible /v1/chat/completions endpoint.

Usage:
    python test_openai_chat.py --api-key sk-... \\
        [--base-url http://127.0.0.1:8000] \\
        [--model qwen2.5-vl-7b-instruct]

The script prints the JSON response to stdout and exits with a non-zero code on error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QueryLake OpenAI-compatible chat tester")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Bearer token (QueryLake API key)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the QueryLake server (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-vl-7b-instruct",
        help="Model identifier to request (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum output tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    return parser.parse_args()


def build_payload(model: str, temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": "Give me three bullet points about the benefits of solar power.",
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def run_request(base_url: str, api_key: str, payload: dict, timeout: float) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.model, args.temperature, args.max_tokens)

    try:
        data = run_request(args.base_url, args.api_key, payload, args.timeout)
    except requests.Timeout:
        print("Request timed out.", file=sys.stderr)
        return 2
    except requests.HTTPError as exc:
        print(f"HTTP error {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 3
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 4

    json.dump(data, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
