"""Serve the local VDN8 results page and its videos with HTTP byte ranges."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_vdn8_four_model_72_20260929")
VIDEOS = Path("/autodl-fs/data/h3_outputs/comfyui_vdn8_four_model_72_20260929")
EXACT_ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_vdn8_bsa_exact_scheduled_20260929")
EXACT_VIDEOS = Path("/autodl-fs/data/h3_outputs/comfyui_vdn8_bsa_exact_scheduled_20260929")


def resolve_route(route: str) -> Path | None:
    route = unquote(urlsplit(route).path)
    if route in ("/bsa-exact", "/bsa-exact/"):
        return EXACT_ROOT / "index.html"
    if route.startswith("/bsa-exact/"):
        relative = route.removeprefix("/bsa-exact/")
        if relative in ("index.html", "results.csv", "report.md",
                        "old_vs_new_reference_deltas.csv"):
            return EXACT_ROOT / relative
        if relative.startswith(("posters/", "videos/")) and relative.endswith((".jpg", ".mp4")):
            path = (EXACT_ROOT / relative).resolve()
            allowed = (EXACT_ROOT, EXACT_VIDEOS, VIDEOS, ROOT / "posters")
            if any(path.is_relative_to(base.resolve()) for base in allowed):
                return path
        return None
    if route == "/":
        return ROOT / "index.html"
    if route in ("/index.html", "/results.csv", "/report.md"):
        return ROOT / route.lstrip("/")
    for prefix, base, suffix in (("/posters/", ROOT / "posters", ".jpg"),
                                 ("/videos/", VIDEOS, ".mp4")):
        if route.startswith(prefix) and route.endswith(suffix):
            path = (base / route.removeprefix(prefix)).resolve()
            if path.is_relative_to(base.resolve()):
                return path
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.respond(head=False)

    def do_HEAD(self) -> None:
        self.respond(head=True)

    def respond(self, *, head: bool) -> None:
        path = resolve_route(self.path)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range)
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(0, size - int(right))
            else:
                start = size
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if request_range else 200)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".html", ".md", ".csv"):
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if request_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6008)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {ROOT / 'index.html'} on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
