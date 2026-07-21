"""Measurement fuser to combine eyewear measurements from front, side, top, and perspective views."""

from __future__ import annotations
import numpy as np
from backend.models import Measurements, FrameMaterial, FrameShape


class MeasurementFuser:
    """Fuses measurements from multiple camera angles into a single unified measurement set."""

    def fuse(self, views: list[tuple[str, Measurements]]) -> Measurements:
        if not views:
            raise ValueError("Cannot fuse empty list of measurements")

        # Group measurements by view type
        by_view: dict[str, list[Measurements]] = {}
        for view_type, m in views:
            by_view.setdefault(view_type, []).append(m)

        # Helper to average a float attribute across a list of Measurements
        def avg_attr(ms_list: list[Measurements], attr: str) -> float:
            values = [getattr(m, attr) for m in ms_list if getattr(m, attr) is not None]
            return float(np.mean(values)) if values else 0.0

        # 1. Determine baseline front measurements
        front_ms = by_view.get("front", [])
        any_ms = [m for ms_list in by_view.values() for m in ms_list]
        base_ms = front_ms[0] if front_ms else any_ms[0]

        # 2. Fuse frame width, lens width, lens height, bridge width
        # Prefer front view, average if multiple front views are uploaded
        if front_ms:
            frame_width = avg_attr(front_ms, "frame_width")
            lens_width = avg_attr(front_ms, "lens_width")
            lens_height = avg_attr(front_ms, "lens_height")
            bridge_width = avg_attr(front_ms, "bridge_width")
        else:
            frame_width = avg_attr(any_ms, "frame_width")
            lens_width = avg_attr(any_ms, "lens_width")
            lens_height = avg_attr(any_ms, "lens_height")
            bridge_width = avg_attr(any_ms, "bridge_width")

        # 3. Fuse temple length and temple curve angle (Prefer side view)
        side_ms = by_view.get("side", [])
        if side_ms:
            temple_length = avg_attr(side_ms, "temple_length")
            temple_curve_angle = avg_attr(side_ms, "temple_curve_angle")
        else:
            # Fallback to default estimation from frame width
            temple_length = frame_width * 0.97
            temple_curve_angle = 28.0

        # 4. Fuse rim thickness (Prefer top view)
        top_ms = by_view.get("top", [])
        if top_ms:
            rim_thickness = avg_attr(top_ms, "rim_thickness")
        else:
            # Fallback estimation
            rim_thickness = max(0.8, frame_width * 0.008)

        # 5. Handle nose pads, material, shape, color, lens_color, lens_opacity
        # Pick from front view if available, otherwise baseline
        nose_pads = base_ms.nose_pads
        material = base_ms.material
        shape = base_ms.shape
        color = base_ms.color
        
        # Merge lens parameters if any view has them
        lens_color = next((m.lens_color for m in any_ms if m.lens_color is not None), None)
        lens_opacity = next((m.lens_opacity for m in any_ms if m.lens_opacity is not None), None)

        return Measurements(
            frame_width=round(frame_width, 1),
            lens_width=round(lens_width, 1),
            lens_height=round(lens_height, 1),
            bridge_width=round(bridge_width, 1),
            temple_length=round(temple_length, 1),
            rim_thickness=round(rim_thickness, 2),
            material=material,
            shape=shape,
            nose_pads=nose_pads,
            temple_curve_angle=round(temple_curve_angle, 1),
            nose_pad_distance=round(bridge_width * 0.6, 1) if nose_pads else None,
            nose_pad_angle=15.0 if nose_pads else None,
            nose_pad_height=3.0 if nose_pads else None,
            color=color,
            lens_color=lens_color,
            lens_opacity=lens_opacity,
        )
