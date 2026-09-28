import gzip
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "slice_api", Path(__file__).resolve().parents[2] / "api" / "slice.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def test_preview_returns_thinned_layers():
    status, headers, body = api.handle({"mode": "preview", "wall": {"height": 40}})
    data = json.loads(body)
    assert status == 200
    assert data["total_layers"] == 200
    assert len(data["layers"]) <= api.PREVIEW_LAYERS + 1
    assert data["layers"][-1]["index"] == 199
    assert all(len(p) % 3 == 0 for layer in data["layers"] for p in layer["paths"])


def test_gcode_is_gzipped():
    status, headers, body = api.handle({"mode": "gcode", "wall": {"shape": "line", "height": 5}})
    assert headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(body).startswith(b"; Non-planar")
    assert json.loads(headers["X-Summary"])["layers"] == 25


@pytest.mark.parametrize("body", [
    {"wall": {"height": "abc"}},
    {"wall": {"shape": "square"}},
    {"wall": {"height": 150, "layer_height": 0.08, "radius": 100}},
    {"mode": "nope"},
])
def test_bad_requests(body):
    with pytest.raises(api.BadRequest):
        api.handle(body)
