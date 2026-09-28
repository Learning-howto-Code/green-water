"""Interactive matplotlib window: sliders on the right, 3D toolpath preview on the left."""

from __future__ import annotations

import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .gcode import PrinterParams, build_gcode
from .geometry import SHAPES, WAVEFORMS, WallParams, generate_layers, wall_stats

# (field, label, min, max, step)
SLIDERS = [
    ("height", "Height (mm)", 2, 150, 0.5),
    ("radius", "Radius (mm)", 5, 100, 0.5),
    ("length", "Length (mm)", 10, 200, 1),
    ("layer_height", "Layer height", 0.08, 0.4, 0.02),
    ("line_width", "Line width", 0.3, 1.0, 0.05),
    ("perimeters", "Perimeters", 1, 6, 1),
    ("z_amplitude", "Z wave amp", 0, 6, 0.1),
    ("z_wavelength", "Z wavelength", 2, 100, 0.5),
    ("z_phase_shift", "Z phase/layer", -20, 20, 0.5),
    ("z_ramp_layers", "Z ramp layers", 1, 100, 1),
    ("xy_amplitude", "XY wave amp", 0, 6, 0.1),
    ("xy_wavelength", "XY wavelength", 2, 100, 0.5),
    ("xy_phase_shift", "XY phase/layer", -20, 20, 0.5),
]

MAX_PREVIEW_LAYERS = 90


class SlicerApp:
    def __init__(self, wall: WallParams, printer: PrinterParams, output: str):
        self.wall = wall
        self.printer = printer
        self.output = output
        self.message = ""

        self.fig = plt.figure("Non-planar wave wall slicer", figsize=(14, 8.5))
        self.ax = self.fig.add_axes([0.0, 0.0, 0.58, 0.94], projection="3d")
        self.sliders: dict[str, Slider] = {}

        left, width, h, top = 0.70, 0.22, 0.026, 0.95
        for n, (name, label, lo, hi, step) in enumerate(SLIDERS):
            ax = self.fig.add_axes([left, top - n * (h + 0.012), width, h])
            s = Slider(ax, label, lo, hi, valinit=getattr(wall, name), valstep=step)
            s.on_changed(self._on_change)
            self.sliders[name] = s

        y = top - len(SLIDERS) * (h + 0.012) - 0.11
        self.shape_radio = self._radio([0.60, y, 0.08, 0.1], "Shape", SHAPES, wall.shape)
        self.zwave_radio = self._radio([0.70, y, 0.09, 0.1], "Z waveform", WAVEFORMS, wall.z_waveform)
        self.xywave_radio = self._radio([0.81, y, 0.09, 0.1], "XY waveform", WAVEFORMS, wall.xy_waveform)

        cax = self.fig.add_axes([0.60, y - 0.07, 0.12, 0.055])
        self.check = CheckButtons(cax, ["Flat top"], [wall.flat_top])
        self.check.on_clicked(self._on_change)

        bax = self.fig.add_axes([0.75, y - 0.07, 0.15, 0.05])
        self.export_btn = Button(bax, "Export G-code")
        self.export_btn.on_clicked(self._export)

        self.stats_text = self.fig.text(0.60, 0.02, "", family="monospace", fontsize=9, va="bottom")
        self.redraw()

    def _radio(self, rect, title, options, active):
        ax = self.fig.add_axes(rect)
        ax.set_title(title, fontsize=9)
        radio = RadioButtons(ax, options, active=options.index(active))
        radio.on_clicked(self._on_change)
        return radio

    def current_wall(self) -> WallParams:
        values = {name: s.val for name, s in self.sliders.items()}
        values["perimeters"] = int(values["perimeters"])
        values["z_ramp_layers"] = int(values["z_ramp_layers"])
        return replace(
            self.wall,
            **values,
            shape=self.shape_radio.value_selected,
            z_waveform=self.zwave_radio.value_selected,
            xy_waveform=self.xywave_radio.value_selected,
            flat_top=self.check.get_status()[0],
        )

    def _on_change(self, _):
        self.message = ""
        self.redraw()

    def redraw(self):
        wall = self.current_wall()
        try:
            layers = generate_layers(wall)
        except ValueError as err:
            self.stats_text.set_text(f"Invalid settings: {err}")
            self.fig.canvas.draw_idle()
            return
        stats = wall_stats(wall, layers)

        step = max(1, len(layers) // MAX_PREVIEW_LAYERS)
        shown = layers[::step]
        if shown[-1] is not layers[-1]:
            shown.append(layers[-1])
        segments, colors = [], []
        cmap = plt.get_cmap("viridis")
        for layer in shown:
            c = cmap(layer.index / max(1, len(layers) - 1))
            for path in layer.paths:
                segments.append(path.points)
                colors.append(c)

        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=0.8))
        allpts = np.vstack([s for s in segments])
        mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
        span = float((maxs - mins).max()) / 2 + 1
        mid = (maxs + mins) / 2
        self.ax.set_xlim(mid[0] - span, mid[0] + span)
        self.ax.set_ylim(mid[1] - span, mid[1] + span)
        self.ax.set_zlim(0, 2 * span)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_title(f"{wall.shape} wall, {len(layers)} layers (showing every {step})")

        warnings = []
        if stats.min_thickness < 0.5 * wall.layer_height:
            warnings.append("! very thin spots: lower phase shift or amplitude")
        if stats.max_thickness > 0.8 * wall.line_width:
            warnings.append("! thick spots may under-extrude")
        if stats.max_slope_deg > 30:
            warnings.append("! steep slope: nozzle may hit the wall")
        text = (
            f"layers            {stats.layers}\n"
            f"layer thickness   {stats.min_thickness:.3f} - {stats.max_thickness:.3f} mm\n"
            f"max slope         {stats.max_slope_deg:.1f} deg\n"
            f"Z waves / XY waves {stats.z_waves:.2f} / {stats.xy_waves:.2f}\n"
            f"effective ramp    {stats.ramp_layers} layers\n"
        )
        text += "\n".join(warnings)
        if self.message:
            text += "\n" + self.message
        self.stats_text.set_text(text)
        self.fig.canvas.draw_idle()

    def _export(self, _):
        wall = self.current_wall()
        try:
            gcode, summary = build_gcode(wall, self.printer)
        except ValueError as err:
            self.message = f"Export failed: {err}"
        else:
            with open(self.output, "w") as fh:
                fh.write(gcode)
            self.message = (
                f"Saved {os.path.basename(self.output)}\n"
                f"  {summary['filament_mm'] / 1000:.2f} m / {summary['filament_g']:.1f} g filament, "
                f"~{summary['time_min']:.0f} min"
            )
            print(f"Saved {os.path.abspath(self.output)}")
        self.redraw()


def run(wall: WallParams, printer: PrinterParams, output: str) -> None:
    app = SlicerApp(wall, printer, output)  # noqa: F841 (keep widgets alive)
    plt.show()
