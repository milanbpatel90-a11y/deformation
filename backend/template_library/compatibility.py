"""Public measurement constraints shared by suggestions and generation."""
import json
from pathlib import Path


class MeasurementCompatibilityError(ValueError):
    """Safe, actionable input error suitable for displaying to a user."""
    def __init__(self, message, ranges=None):
        super().__init__(message)
        self.ranges = ranges or {}


def dependent_ranges(measurements, base_ranges=None):
    """Ranges for changing ONE field while keeping the other inputs fixed.

    FW >= 2*LW + BW + 2*RT reserves a positive outer margin per side.
    The nominal bridge control is defined as the horizontal lens-box gap.
    """
    m = measurements
    ranges = {key: dict(value) for key, value in (base_ranges or {}).items()}
    limits = {
        "frame_width": (2*m.lens_width + m.bridge_width + 2*m.rim_thickness, 300.0),
        "lens_width": (0.001, (m.frame_width-m.bridge_width-2*m.rim_thickness)/2),
        "bridge_width": (0.001, m.frame_width-2*m.lens_width-2*m.rim_thickness),
        "rim_thickness": (0.001, min(20.0, (m.frame_width-2*m.lens_width-m.bridge_width)/2)),
    }
    for field, (low, high) in limits.items():
        existing = ranges.get(field, {"min": 0.001, "max": 300.0})
        lo, hi = max(low, existing["min"]), min(high, existing["max"])
        ranges[field] = {"min": round(lo, 8), "max": round(hi, 8), "feasible": lo <= hi + 1e-9}
    return ranges


def validate_combination(measurements, base_ranges=None):
    m = measurements
    required = 2*m.lens_width + m.bridge_width + 2*m.rim_thickness
    if m.frame_width + 1e-8 < required:
        raise MeasurementCompatibilityError(
            f"Frame width must be at least {required:g} mm: "
            f"2 × lens width ({m.lens_width:g}) + bridge width ({m.bridge_width:g}) "
            f"+ 2 × rim thickness ({m.rim_thickness:g}); received {m.frame_width:g} mm.",
            dependent_ranges(m, base_ranges))


def measurement_ranges(template):
    if template.deformation_mode != "basis":
        return {}
    path = Path(template.glb_path).parents[1] / "deformation/basis_metadata.json"
    return {p["name"]: {"min": p["min"], "max": p["max"]}
            for p in json.loads(path.read_text(encoding="utf-8"))["parameters"]}


def incompatibilities(template, measurements):
    return [f'{name.replace("_", " ").title()}: {getattr(measurements, name):g} mm; supported range {bounds["min"]:g}–{bounds["max"]:g} mm'
            for name, bounds in measurement_ranges(template).items()
            if not bounds["min"] <= getattr(measurements, name) <= bounds["max"]]
