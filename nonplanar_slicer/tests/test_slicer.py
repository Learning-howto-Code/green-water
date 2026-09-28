import re

import numpy as np
import pytest

from nonplanar_slicer import PrinterParams, WallParams, build_gcode, generate_layers, wall_stats


@pytest.mark.parametrize("shape", ["circle", "line"])
def test_first_layer_flat_and_thickness_positive(shape):
    wall = WallParams(shape=shape, height=10, z_amplitude=2, perimeters=2)
    layers = generate_layers(wall)
    first = layers[0].paths[0].points
    assert np.allclose(first[:, 2], wall.first_layer_height)
    stats = wall_stats(wall, layers)
    assert stats.min_thickness >= 0.5 * wall.layer_height - 1e-9


def test_flat_top_ends_at_height():
    wall = WallParams(height=12, z_amplitude=2)
    top = generate_layers(wall)[-1].paths[0].points
    assert np.allclose(top[:, 2], top[0, 2])
    assert abs(top[0, 2] - wall.height) < wall.layer_height


def test_circle_wave_closes():
    wall = WallParams(shape="circle", z_wavelength=17.3, xy_wavelength=7.1, height=5)
    pts = generate_layers(wall)[-5].paths[0].points
    assert np.allclose(pts[0], pts[-1], atol=1e-6)


def test_line_paths_chain_without_travel():
    wall = WallParams(shape="line", perimeters=3, height=3, xy_amplitude=0)
    paths = [p for l in generate_layers(wall) for p in l.paths]
    for a, b in zip(paths, paths[1:]):
        gap = np.hypot(*(b.points[0, :2] - a.points[-1, :2]))
        assert gap <= wall.line_width + 1e-6


def test_gcode_is_nonplanar_and_extrudes():
    wall = WallParams(height=5, z_amplitude=1)
    text, summary = build_gcode(wall, PrinterParams())
    moves = re.findall(r"^G1 X[\d.]+ Y[\d.]+ Z([\d.]+) E([\d.]+)$", text, re.M)
    assert moves
    assert all(float(e) >= 0 for _, e in moves)
    assert summary["filament_mm"] > 0
    # Within a single upper layer Z must vary.
    last_layer = text.split(";LAYER:")[-5]
    zs = {z for z, _ in re.findall(r"^G1 X[\d.]+ Y[\d.]+ Z([\d.]+) E([\d.]+)$", last_layer, re.M)}
    assert len(zs) > 10
