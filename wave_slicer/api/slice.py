"""Vercel Python function: POST /api/slice

Request JSON: {"mode": "preview" | "gcode", "wall": {...}, "printer": {...}}
Keys in "wall" / "printer" match WallParams / PrinterParams fields; missing
keys use the defaults.

* preview -> JSON with thinned-out toolpaths for the 3D view plus stats
* gcode   -> the G-code file (gzip encoded to stay under Vercel's 4.5 MB limit)

Run locally without Vercel:  python api/slice.py   (serves on port 8000)
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nonplanar_slicer.gcode import PrinterParams, build_gcode  # noqa: E402
from nonplanar_slicer.geometry import (  # noqa: E402
    WallParams, generate_layers, layer_count, path_length, wall_stats, wall_warnings,
)

MAX_POINTS = 600_000  # keeps the G-code under the response size limit
MAX_RESPONSE_BYTES = 4_400_000
PREVIEW_LAYERS = 80
PREVIEW_POINTS = 240


class BadRequest(ValueError):
    pass


def _parse(cls, data):
    if not isinstance(data, dict):
        raise BadRequest(f"{cls.__name__} must be an object")
    values = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kind = type(f.default)
        raw = data[f.name]
        try:
            if kind is bool:
                values[f.name] = bool(raw)
            elif kind is int:
                values[f.name] = int(round(float(raw)))
            elif kind is float:
                values[f.name] = float(raw)
                if not math.isfinite(values[f.name]):
                    raise ValueError
            else:
                values[f.name] = str(raw)
        except (TypeError, ValueError):
            raise BadRequest(f"bad value for {f.name}: {raw!r}") from None
    return cls(**values)


def _point_count(wall: WallParams) -> int:
    per_path = math.ceil(path_length(wall) / wall.resolution) + 1
    return layer_count(wall) * wall.perimeters * per_path


def handle(body: dict) -> tuple[int, dict, bytes]:
    """Return (status, headers, body). Kept separate from HTTP for testing."""
    wall = _parse(WallParams, body.get("wall", {}))
    printer = _parse(PrinterParams, body.get("printer", {}))
    try:
        wall.validate()
    except ValueError as err:
        raise BadRequest(str(err)) from None
    if _point_count(wall) > MAX_POINTS:
        raise BadRequest(
            "wall is too big for the web version: raise layer height or resolution, "
            "or use the command line tool"
        )

    layers = generate_layers(wall)
    mode = body.get("mode", "preview")

    if mode == "gcode":
        text, summary = build_gcode(wall, printer, layers)
        data = gzip.compress(text.encode(), compresslevel=6)
        if len(data) > MAX_RESPONSE_BYTES:
            raise BadRequest("G-code is too large to download here: use the command line tool")
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Encoding": "gzip",
            "Content-Disposition": 'attachment; filename="wave_wall.gcode"',
            "X-Summary": json.dumps(summary),
        }
        return 200, headers, data

    if mode != "preview":
        raise BadRequest("mode must be 'preview' or 'gcode'")

    stats = wall_stats(wall, layers)
    step = max(1, math.ceil(len(layers) / PREVIEW_LAYERS))
    shown = layers[::step]
    if shown[-1] is not layers[-1]:
        shown.append(layers[-1])
    out_layers = []
    for layer in shown:
        paths = []
        for path in layer.paths:
            pts = path.points
            k = max(1, math.ceil(len(pts) / PREVIEW_POINTS))
            thin = pts[::k]
            if k > 1 and (len(pts) - 1) % k:
                thin = list(thin) + [pts[-1]]
            paths.append([round(float(v), 2) for p in thin for v in p])
        out_layers.append({"index": layer.index, "paths": paths})

    payload = {
        "layers": out_layers,
        "total_layers": len(layers),
        "stats": asdict(stats),
        "warnings": wall_warnings(wall, stats),
    }
    return 200, {"Content-Type": "application/json"}, json.dumps(payload).encode()


class handler(BaseHTTPRequestHandler):
    def _send(self, status, headers, body: bytes):
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 100_000:
                raise BadRequest("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise BadRequest("request must be a JSON object")
            status, headers, data = handle(body)
        except (BadRequest, json.JSONDecodeError) as err:
            status, headers = 400, {"Content-Type": "application/json"}
            data = json.dumps({"error": str(err)}).encode()
        self._send(status, headers, data)

    def do_GET(self):
        self._send(200, {"Content-Type": "application/json"},
                   json.dumps({"wall": asdict(WallParams()), "printer": asdict(PrinterParams())}).encode())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Slicer API on http://localhost:{port}/api/slice")
    HTTPServer(("", port), handler).serve_forever()
