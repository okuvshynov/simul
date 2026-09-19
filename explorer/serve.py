#!/usr/bin/env python3
"""Tiny web explorer for logs/. Run from anywhere: python3 explorer/serve.py [port]"""

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/explorer/index.html"
        elif self.path == "/api/logs":
            return self.send_json(list_logs())
        return super().do_GET()

    def end_headers(self):
        # logs change while runs are in progress; never let the browser cache them
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet


def list_logs():
    """All logs without their trace, plus the file name to link to."""
    out = []
    for p in sorted(LOGS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue  # partially written file, skip for now
        d.pop("trace", None)
        d["file"] = p.name
        out.append(d)
    return out


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving {LOGS} at http://localhost:{port}/")
    ThreadingHTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
