"""Measurement-driven deformation of template parts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .measurements import Measurements


def deform_template(template: dict[str, Any], measurements: Measurements) -> dict[str, Any]:
    parts = {name: mesh.copy() for name, mesh in template["parts"].items()}
    target_width = measurements.lens_width_mm / 1000.0
    target_height = measurements.lens_height_mm / 1000.0
    for name in ("lens_L", "lens_R"):
        ext = parts[name].extents
        transform = np.eye(4)
        transform[0, 0] = target_width / max(ext[0], 1e-6)
        transform[1, 1] = target_height / max(ext[1], 1e-6)
        parts[name].apply_transform(transform)
    for name in ("front",):
        current = max(parts[name].extents[0], 1e-6)
        transform = np.eye(4); transform[0, 0] = (measurements.total_frame_width_mm / 1000) / current
        parts[name].apply_transform(transform)
    for name in ("temple_L", "temple_R"):
        current = max(parts[name].extents[0], 1e-6)
        transform = np.eye(4); transform[0, 0] = (measurements.temple_length_mm / 1000) / current
        parts[name].apply_transform(transform)
    return {**template, "parts": parts, "measurements": measurements.to_dict()}
