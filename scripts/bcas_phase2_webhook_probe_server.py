#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny webhook probe server for BCAS notification drills.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=1)
    args = parser.parse_args()

    args.capture.parent.mkdir(parents=True, exist_ok=True)

    state = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            state["count"] += 1
            rec = {
                "received_at_unix": time.time(),
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
            with args.capture.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            if state["count"] >= int(args.max_requests):
                self.server.should_exit = True

        def log_message(self, format, *args):
            return

    server = HTTPServer((args.host, args.port), Handler)
    server.should_exit = False
    while not server.should_exit:
        server.handle_request()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
