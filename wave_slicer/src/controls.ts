// Control definitions. Field names match WallParams / PrinterParams in Python.

export type Slider = {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
  only?: "circle" | "line";
};

export type Group = { title: string; hint?: string; sliders: Slider[] };

export const wallGroups: Group[] = [
  {
    title: "Size",
    sliders: [
      { key: "height", label: "Height", min: 2, max: 150, step: 0.5, value: 30, unit: "mm" },
      { key: "radius", label: "Radius", min: 5, max: 100, step: 0.5, value: 25, unit: "mm", only: "circle" },
      { key: "length", label: "Length", min: 10, max: 200, step: 1, value: 80, unit: "mm", only: "line" },
      { key: "layer_height", label: "Layer height", min: 0.08, max: 0.4, step: 0.02, value: 0.2, unit: "mm" },
      { key: "line_width", label: "Line width", min: 0.3, max: 1, step: 0.05, value: 0.45, unit: "mm" },
      { key: "perimeters", label: "Perimeters", min: 1, max: 6, step: 1, value: 1 },
    ],
  },
  {
    title: "Z wave",
    hint: "Bends each layer up and down: the non-planar part.",
    sliders: [
      { key: "z_amplitude", label: "Amplitude", min: 0, max: 6, step: 0.1, value: 1.5, unit: "mm" },
      { key: "z_wavelength", label: "Wavelength", min: 2, max: 100, step: 0.5, value: 20, unit: "mm" },
      { key: "z_phase_shift", label: "Phase / layer", min: -20, max: 20, step: 0.5, value: 0, unit: "°" },
      { key: "z_ramp_layers", label: "Ramp layers", min: 1, max: 100, step: 1, value: 10 },
    ],
  },
  {
    title: "XY wave",
    hint: "Ripples the wall sideways. Phase shift makes diagonal patterns.",
    sliders: [
      { key: "xy_amplitude", label: "Amplitude", min: 0, max: 6, step: 0.1, value: 1, unit: "mm" },
      { key: "xy_wavelength", label: "Wavelength", min: 2, max: 100, step: 0.5, value: 10, unit: "mm" },
      { key: "xy_phase_shift", label: "Phase / layer", min: -20, max: 20, step: 0.5, value: 3, unit: "°" },
    ],
  },
];

export const printerFields: Slider[] = [
  { key: "bed_center_x", label: "Bed centre X", min: 0, max: 500, step: 0.5, value: 110, unit: "mm" },
  { key: "bed_center_y", label: "Bed centre Y", min: 0, max: 500, step: 0.5, value: 110, unit: "mm" },
  { key: "nozzle_temp", label: "Nozzle temp", min: 150, max: 300, step: 1, value: 210, unit: "°C" },
  { key: "bed_temp", label: "Bed temp", min: 0, max: 120, step: 1, value: 60, unit: "°C" },
  { key: "print_speed", label: "Print speed", min: 5, max: 150, step: 1, value: 25, unit: "mm/s" },
  { key: "first_layer_speed", label: "First layer speed", min: 5, max: 60, step: 1, value: 15, unit: "mm/s" },
  { key: "travel_speed", label: "Travel speed", min: 20, max: 300, step: 5, value: 120, unit: "mm/s" },
  { key: "max_z_speed", label: "Max Z speed", min: 1, max: 50, step: 0.5, value: 10, unit: "mm/s" },
  { key: "filament_diameter", label: "Filament Ø", min: 1.5, max: 3, step: 0.05, value: 1.75, unit: "mm" },
  { key: "flow", label: "Flow", min: 0.5, max: 1.5, step: 0.01, value: 1 },
  { key: "retract_length", label: "Retraction", min: 0, max: 8, step: 0.1, value: 0.8, unit: "mm" },
  { key: "fan_percent", label: "Fan", min: 0, max: 100, step: 5, value: 100, unit: "%" },
];
