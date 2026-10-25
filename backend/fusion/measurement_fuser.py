"""Measurement fuser — combine multi-view measurements into one consistent set."""

from __future__ import annotations

import numpy as np

from backend.models import Measurements


class MeasurementFuser:
    """Fuses measurements from front, side, top, and perspective views."""

    VIEW_WEIGHTS = {
        "front": 1.0,
        "side": 0.85,
        "top": 0.7,
        "three_quarter": 0.6,
        "perspective": 0.5,
        "unknown": 0.4,
    }

    def fuse(self, views: list[tuple[str, Measurements]]) -> Measurements:
        if not views:
            raise ValueError("Cannot fuse empty list of measurements")

        by_view: dict[str, list[Measurements]] = {}
        for view_type, m in views:
            key = (view_type or "unknown").lower().strip()
            by_view.setdefault(key, []).append(m)

        any_ms = [m for ms_list in by_view.values() for m in ms_list]
        front_ms = by_view.get("front", [])
        base_ms = front_ms[0] if front_ms else any_ms[0]

        def weighted_avg(attr: str, preferred: list[str] | None = None) -> float | None:
            # Physical dimensions are view-dependent. If a suitable view exists,
            # measurements from that view are authoritative for this attribute;
            # other viewpoints must not outvote it by chance or by count.
            eligible_views = preferred if preferred and any(v in by_view for v in preferred) else list(by_view)
            pairs: list[tuple[float, float]] = []
            for vtype in eligible_views:
                w = self.VIEW_WEIGHTS.get(vtype, 0.4)
                for m in by_view.get(vtype, []):
                    val = getattr(m, attr, None)
                    if val is None:
                        continue
                    try:
                        numeric = float(val)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(numeric):
                        pairs.append((numeric, w))
            if not pairs:
                return None
            values = np.array([p[0] for p in pairs], dtype=float)
            weights = np.array([p[1] for p in pairs], dtype=float)
            if len(values) >= 3:
                median = float(np.median(values))
                deviations = np.abs(values - median)
                mad = float(np.median(deviations))
                if mad == 0.0:
                    keep = np.isclose(values, median, atol=1e-6, rtol=0.0)
                else:
                    keep = deviations <= 3.0 * 1.4826 * mad
                if np.any(keep):
                    values, weights = values[keep], weights[keep]
            return float(np.average(values, weights=weights))

        frame_width = weighted_avg("frame_width", ["front"]) or base_ms.frame_width
        lens_width = weighted_avg("lens_width", ["front"]) or base_ms.lens_width
        lens_height = weighted_avg("lens_height", ["front"]) or base_ms.lens_height
        bridge_width = weighted_avg("bridge_width", ["front"]) or base_ms.bridge_width

        temple_length = weighted_avg("temple_length", ["side"])
        temple_curve_angle = weighted_avg("temple_curve_angle", ["side"])
        if temple_length is None:
            temple_length = frame_width * 0.97
        if temple_curve_angle is None:
            temple_curve_angle = 28.0

        rim_thickness = weighted_avg("rim_thickness", ["top"])
        if rim_thickness is None:
            rim_thickness = max(0.8, frame_width * 0.008)

        nose_pads = base_ms.nose_pads
        material = base_ms.material
        shape = base_ms.shape
        color = base_ms.color
        lens_color = next((m.lens_color for m in any_ms if m.lens_color is not None), None)
        lens_opacity = next((m.lens_opacity for m in any_ms if m.lens_opacity is not None), None)

        frame_width = float(np.clip(frame_width, 100.0, 180.0))
        lens_width = float(np.clip(lens_width, 28.0, 70.0))
        lens_height = float(np.clip(lens_height, 18.0, 58.0))
        bridge_width = float(np.clip(bridge_width, 8.0, 30.0))
        temple_length = float(np.clip(temple_length, 115.0, 160.0))
        rim_thickness = float(np.clip(rim_thickness, 0.5, 4.0))

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
            temple_curve_angle=round(float(temple_curve_angle), 1),
            nose_pad_distance=round(bridge_width * 0.6, 1) if nose_pads else None,
            nose_pad_angle=15.0 if nose_pads else None,
            nose_pad_height=3.0 if nose_pads else None,
            color=color,
            lens_color=lens_color,
            lens_opacity=lens_opacity,
        )
