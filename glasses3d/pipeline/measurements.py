"""Convert normalized landmarks into millimetres."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Measurements:
    lens_width_mm: float
    lens_height_mm: float
    bridge_width_mm: float
    total_frame_width_mm: float
    temple_length_mm: float
    frame_thickness_mm: float
    pantoscopic_tilt_deg: float = 8.0

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def extract_measurements(landmarks: dict[str, Any], frame_width_mm: float | None = None) -> Measurements:
    front = landmarks["front"]
    scale = float(frame_width_mm or 140.0)
    frame_width_norm = abs(front["hinge_R"][0] - front["hinge_L"][0]) or 0.7
    mm_per_norm = scale / frame_width_norm
    lens_width = abs(front["lens_outer_L"][0] - front["lens_inner_L"][0]) * mm_per_norm
    lens_height = abs(front["frame_bottom"][1] - front["frame_top"][1]) * mm_per_norm * 0.72
    bridge = abs(front["lens_inner_R"][0] - front["lens_inner_L"][0]) * mm_per_norm
    side = landmarks.get("side")
    temple = 135.0 if not side else abs(side["temple_end"][0] - side["temple_start"][0]) * 135.0
    return Measurements(max(lens_width, 25.0), max(lens_height, 18.0), max(bridge, 8.0), scale, temple, max(1.5, lens_width * 0.035), float(side["pantoscopic_tilt_deg"] if side else 8.0))
