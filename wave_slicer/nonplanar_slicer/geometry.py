"""Wall geometry and non-planar layer generation.

A wall is defined by a base path in XY (a straight line or a circle). Every
point on that path is addressed by ``t``, the distance travelled along it.
Two independent waves can be applied:

* the **Z wave** bends each layer up and down along the path, which is what
  makes the print non-planar. The layers stay parallel to each other, so the
  wall is built from stacked wavy ribbons instead of flat slices.
* the **XY wave** pushes the wall sideways (along the path normal), giving a
  corrugated surface. Shifting its phase every layer turns the ripples into
  diagonal or twisting patterns.

The first layer is always flat so it sticks to the bed. The Z wave amplitude
ramps in over a number of layers, and can ramp back out for a flat top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SHAPES = ("circle", "line")
WAVEFORMS = ("sine", "triangle")


@dataclass
class WallParams:
    shape: str = "circle"  # "circle" or "line"
    length: float = 80.0  # mm, used by "line"
    radius: float = 25.0  # mm, used by "circle"
    height: float = 30.0  # mm
    layer_height: float = 0.2  # mm
    first_layer_height: float = 0.25  # mm
    line_width: float = 0.45  # mm
    perimeters: int = 1  # side-by-side lines per layer

    # Vertical (non-planar) wave
    z_amplitude: float = 1.5  # mm
    z_wavelength: float = 20.0  # mm along the path
    z_phase_shift: float = 0.0  # degrees added per layer
    z_ramp_layers: int = 10  # layers used to fade the wave in (and out)
    flat_top: bool = True
    z_waveform: str = "sine"

    # Lateral (XY) wave
    xy_amplitude: float = 1.0  # mm
    xy_wavelength: float = 10.0  # mm along the path
    xy_phase_shift: float = 0.0  # degrees added per layer
    xy_waveform: str = "sine"

    resolution: float = 0.5  # mm between points along the path

    def validate(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        for name in ("z_waveform", "xy_waveform"):
            if getattr(self, name) not in WAVEFORMS:
                raise ValueError(f"{name} must be one of {WAVEFORMS}")
        positive = ("length", "radius", "height", "layer_height",
                    "first_layer_height", "line_width", "z_wavelength",
                    "xy_wavelength", "resolution")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.perimeters < 1:
            raise ValueError("perimeters must be >= 1")
        if self.z_amplitude < 0 or self.xy_amplitude < 0:
            raise ValueError("amplitudes must be >= 0")
        if self.height < self.first_layer_height:
            raise ValueError("height must be at least the first layer height")


@dataclass
class Path:
    """One continuous extrusion: points (N x 3) and local layer thickness (N)."""
    points: np.ndarray
    thickness: np.ndarray


@dataclass
class Layer:
    index: int
    paths: list[Path] = field(default_factory=list)


def _wave(phase: np.ndarray, waveform: str) -> np.ndarray:
    """Periodic wave in [-1, 1] with period 2*pi."""
    if waveform == "triangle":
        return (2.0 / math.pi) * np.arcsin(np.sin(phase))
    return np.sin(phase)


def path_length(p: WallParams) -> float:
    return 2.0 * math.pi * p.radius if p.shape == "circle" else p.length


def effective_wavelength(p: WallParams, wavelength: float) -> float:
    """On a closed circle the wave must repeat a whole number of times."""
    if p.shape != "circle":
        return wavelength
    total = path_length(p)
    count = max(1, round(total / wavelength))
    return total / count


def layer_count(p: WallParams) -> int:
    return int(round((p.height - p.first_layer_height) / p.layer_height)) + 1


def effective_ramp_layers(p: WallParams) -> int:
    """Ramp long enough that amplitude never changes by more than a quarter layer."""
    if p.z_amplitude == 0:
        return 1
    needed = math.ceil(p.z_amplitude / (0.25 * p.layer_height))
    return max(1, p.z_ramp_layers, needed)


def amplitude_envelope(p: WallParams) -> np.ndarray:
    """Z-wave amplitude for each layer. Layer 0 is always flat."""
    n = layer_count(p)
    ramp = effective_ramp_layers(p)
    idx = np.arange(n, dtype=float)
    env = np.clip(idx / ramp, 0.0, 1.0)
    if p.flat_top:
        env = np.minimum(env, np.clip((n - 1 - idx) / ramp, 0.0, 1.0))
    return p.z_amplitude * env


def _base_path(p: WallParams, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base XY positions and outward unit normals, centred on (0, 0)."""
    if p.shape == "circle":
        theta = t / p.radius
        normal = np.column_stack((np.cos(theta), np.sin(theta)))
        return p.radius * normal, normal
    xy = np.column_stack((t - p.length / 2.0, np.zeros_like(t)))
    normal = np.column_stack((np.zeros_like(t), np.ones_like(t)))
    return xy, normal


def sample_t(p: WallParams) -> np.ndarray:
    total = path_length(p)
    n = max(2, int(math.ceil(total / p.resolution)) + 1)
    return np.linspace(0.0, total, n)


def layer_z(p: WallParams, t: np.ndarray, layer: int, env: np.ndarray) -> np.ndarray:
    """Nozzle Z for a layer along the path. Layer -1 is the bed (z = 0)."""
    if layer < 0:
        return np.zeros_like(t)
    base = p.first_layer_height + layer * p.layer_height
    wl = effective_wavelength(p, p.z_wavelength)
    phase = 2.0 * math.pi * t / wl + math.radians(p.z_phase_shift) * layer
    return base + env[layer] * _wave(phase, p.z_waveform)


def layer_xy_offset(p: WallParams, t: np.ndarray, layer: int) -> np.ndarray:
    if p.xy_amplitude == 0:
        return np.zeros_like(t)
    wl = effective_wavelength(p, p.xy_wavelength)
    phase = 2.0 * math.pi * t / wl + math.radians(p.xy_phase_shift) * layer
    return p.xy_amplitude * _wave(phase, p.xy_waveform)


def generate_layers(p: WallParams) -> list[Layer]:
    p.validate()
    t = sample_t(p)
    base_xy, normal = _base_path(p, t)
    env = amplitude_envelope(p)
    layers: list[Layer] = []
    # Perimeters are centred on the wave line so the wall stays symmetric.
    offsets = [(k - (p.perimeters - 1) / 2.0) * p.line_width for k in range(p.perimeters)]

    prev_z = layer_z(p, t, -1, env)
    for i in range(layer_count(p)):
        z = layer_z(p, t, i, env)
        thickness = z - prev_z
        shift = layer_xy_offset(p, t, i)
        layer = Layer(index=i)

        # Alternate perimeter order each layer so line walls zig-zag without travel.
        order = range(p.perimeters) if i % 2 == 0 else reversed(range(p.perimeters))
        for n, k in enumerate(order):
            xy = base_xy + (shift + offsets[k])[:, None] * normal
            pts = np.column_stack((xy, z))
            th = thickness.copy()
            if p.shape == "line" and (i * p.perimeters + n) % 2 == 1:
                pts, th = pts[::-1], th[::-1]
            layer.paths.append(Path(points=pts, thickness=th))
        layers.append(layer)
        prev_z = z
    return layers


@dataclass
class WallStats:
    layers: int
    min_thickness: float
    max_thickness: float
    max_slope_deg: float
    z_waves: float
    xy_waves: float
    ramp_layers: int


def wall_stats(p: WallParams, layers: list[Layer]) -> WallStats:
    min_th = min(float(path.thickness.min()) for l in layers for path in l.paths)
    max_th = max(float(path.thickness.max()) for l in layers for path in l.paths)
    max_slope = 0.0
    for layer in layers:
        pts = layer.paths[0].points
        d = np.diff(pts, axis=0)
        horiz = np.hypot(d[:, 0], d[:, 1])
        slope = np.degrees(np.arctan2(np.abs(d[:, 2]), horiz))
        max_slope = max(max_slope, float(slope.max()))
    total = path_length(p)
    return WallStats(
        layers=len(layers),
        min_thickness=min_th,
        max_thickness=max_th,
        max_slope_deg=max_slope,
        z_waves=total / effective_wavelength(p, p.z_wavelength),
        xy_waves=total / effective_wavelength(p, p.xy_wavelength),
        ramp_layers=effective_ramp_layers(p),
    )


def wall_warnings(p: WallParams, stats: WallStats) -> list[str]:
    warnings = []
    if stats.min_thickness < 0.5 * p.layer_height:
        warnings.append("very thin spots: lower phase shift or amplitude")
    if stats.max_thickness > 0.8 * p.line_width:
        warnings.append("thick spots may under-extrude")
    if stats.max_slope_deg > 30:
        warnings.append("steep slope: nozzle may hit the wall")
    return warnings
