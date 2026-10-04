"""Robust multi-view measurement fusion used by the real deformation pipeline."""

from __future__ import annotations

from typing import Sequence

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.multiview.fuse_measurements import robust_estimate


class MeasurementFuser:
    """Fuse view-specific measurements using weighted median + MAD rejection."""

    VIEW_WEIGHTS = {"front": 1.0, "perspective": 0.8, "side": 0.6, "top": 0.6}

    def _robust(self, views: list[tuple[str, Measurements]], attr: str) -> float:
        values = [float(getattr(measurement, attr)) for _, measurement in views]
        weights = [self.VIEW_WEIGHTS.get(view_type, 0.5) for view_type, _ in views]
        estimate, _ = robust_estimate(values, weights)
        return estimate

    def fuse(self, views: list[tuple[str, Measurements]]) -> Measurements:
        if not views:
            raise ValueError("Cannot fuse empty list of measurements")

        by_view: dict[str, list[Measurements]] = {}
        for view_type, measurement in views:
            by_view.setdefault(view_type, []).append(measurement)

        front_views = [(view, measurement) for view, measurement in views if view in {"front", "perspective"}]
        all_views = list(views)
        geometry_views = front_views or all_views

        base = by_view.get("front", by_view.get("perspective", [all_views[0][1]]))[0]

        frame_width = self._robust(geometry_views, "frame_width")
        lens_width = self._robust(geometry_views, "lens_width")
        lens_height = self._robust(geometry_views, "lens_height")
        bridge_width = self._robust(geometry_views, "bridge_width")

        side_views = [("side", measurement) for measurement in by_view.get("side", [])]
        side_source = side_views or all_views
        temple_length = self._robust(side_source, "temple_length")
        temple_curve_angle = self._robust(side_source, "temple_curve_angle")

        top_views = [("top", measurement) for measurement in by_view.get("top", [])]
        rim_source = top_views or all_views
        rim_thickness = self._robust(rim_source, "rim_thickness")

        lens_color = next(
            (measurement.lens_color for _, measurement in all_views if measurement.lens_color is not None),
            None,
        )
        lens_opacity = next(
            (measurement.lens_opacity for _, measurement in all_views if measurement.lens_opacity is not None),
            None,
        )

        return Measurements(
            frame_width=round(frame_width, 1),
            lens_width=round(lens_width, 1),
            lens_height=round(lens_height, 1),
            bridge_width=round(bridge_width, 1),
            temple_length=round(temple_length, 1),
            rim_thickness=round(rim_thickness, 2),
            material=base.material,
            shape=base.shape,
            nose_pads=base.nose_pads,
            temple_curve_angle=round(temple_curve_angle, 1),
            nose_pad_distance=round(bridge_width * 0.6, 1) if base.nose_pads else None,
            nose_pad_angle=15.0 if base.nose_pads else None,
            nose_pad_height=3.0 if base.nose_pads else None,
            color=base.color,
            lens_color=lens_color,
            lens_opacity=lens_opacity,
        )
