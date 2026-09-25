"""Public measurement constraints shared by suggestions and generation."""
import json
from pathlib import Path


class MeasurementCompatibilityError(ValueError):
    """Safe, actionable input error suitable for displaying to a user."""


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
