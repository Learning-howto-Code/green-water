"""Command line entry point.

    python -m nonplanar_slicer                     # open the slider GUI
    python -m nonplanar_slicer --no-gui -o w.gcode --z-amplitude 2 --shape line
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from .gcode import PrinterParams, write_gcode
from .geometry import SHAPES, WAVEFORMS, WallParams

CHOICES = {"shape": SHAPES, "z_waveform": WAVEFORMS, "xy_waveform": WAVEFORMS}


def _add_dataclass_args(parser: argparse.ArgumentParser, cls, title: str) -> None:
    group = parser.add_argument_group(title)
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, bool):
            group.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            group.add_argument(flag, type=type(default), default=default,
                               choices=CHOICES.get(f.name), help=f"default: {default}")


def _build(cls, args):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="nonplanar_slicer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="wave_wall.gcode", help="G-code file to write")
    parser.add_argument("--no-gui", action="store_true", help="write G-code without opening the GUI")
    _add_dataclass_args(parser, WallParams, "wall")
    _add_dataclass_args(parser, PrinterParams, "printer")
    args = parser.parse_args(argv)

    wall, printer = _build(WallParams, args), _build(PrinterParams, args)
    if args.no_gui:
        s = write_gcode(args.output, wall, printer)
        print(f"Wrote {args.output}: {s['layers']} layers, "
              f"thickness {s['min_thickness']:.3f}-{s['max_thickness']:.3f} mm, "
              f"max slope {s['max_slope_deg']:.1f} deg, "
              f"{s['filament_g']:.1f} g, ~{s['time_min']:.0f} min")
        return

    from .gui import run
    run(wall, printer, args.output)


if __name__ == "__main__":
    main()
