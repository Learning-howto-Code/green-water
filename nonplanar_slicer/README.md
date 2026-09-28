# nonplanar_slicer

A small Python G-code generator for **non-planar walls with wave patterns**.
Instead of slicing an STL into flat layers, it builds the wall directly from
parameters, and each layer is a wavy 3D ribbon where X, Y and Z all change
along the path.

## Install and run

```bash
cd nonplanar_slicer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd ..
python -m nonplanar_slicer            # opens the slider GUI
```

Move the sliders, look at the preview, then press **Export G-code**. The file
goes to `wave_wall.gcode` in the current folder (change it with `-o`).

Without the GUI:

```bash
python -m nonplanar_slicer --no-gui -o vase.gcode --shape circle --radius 30 \
    --z-amplitude 2 --z-wavelength 25 --xy-amplitude 1 --xy-phase-shift 4
python -m nonplanar_slicer --help     # every wall and printer option
```

Printer settings (bed centre, temperatures, speeds, retraction) are set on the
command line and used by both modes, e.g.
`--bed-center-x 117.5 --bed-center-y 117.5 --nozzle-temp 215`.
To change the start or end G-code, edit `START_GCODE` / `END_GCODE` in
`gcode.py` with `micro nonplanar_slicer/gcode.py`.

## Wave settings

| Slider | What it does |
| --- | --- |
| Z wave amp / wavelength | Bends every layer up and down. This is the non-planar part. |
| Z phase/layer | Shifts the Z wave each layer. Makes layer thickness vary, so keep it small. |
| Z ramp layers | Layers used to fade the Z wave in from the flat first layer (and out again if **Flat top** is on). The slicer raises it automatically so thickness never changes by more than a quarter layer. |
| XY wave amp / wavelength | Pushes the wall sideways to make a corrugated surface. |
| XY phase/layer | Shifts the ripples each layer: diagonal / twisted patterns. |
| Waveform | Sine or triangle for each wave. |
| Perimeters | Lines placed side by side for a thicker wall. |

On a circle, wavelengths are rounded so a whole number of waves fits around
it and the pattern closes cleanly.

Extrusion is calculated per segment from the real 3D length and the local
layer thickness (the gap to the layer below at that point), so the flow stays
correct where layers are thinner or thicker.

## Things to watch before printing

* **Slope.** The preview reports the steepest slope along a layer. Above
  roughly 30 degrees the side of the nozzle can hit the wall. Lower the Z
  amplitude or use a longer Z wavelength.
* **Z speed.** Every move includes Z, and most printers have slow Z axes
  (often 5–12 mm/s). The firmware slows the whole move down to respect that,
  so non-planar walls print slower than the speed setting suggests.
  Set `--max-z-speed` to your firmware limit for a better time estimate.
* **Thin/thick spots.** The stats panel shows the local layer thickness range
  and warns when it gets very thin or thick.
* Start small (low amplitude, one perimeter) and watch the first print.

## Tests

```bash
pip install pytest
python -m pytest nonplanar_slicer/tests
```
