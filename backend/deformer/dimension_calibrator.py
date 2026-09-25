"""Final physical-dimension calibration for resolved production components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.geometry_units import m_to_mm, mm_to_m


class DimensionCalibrator(BaseDeformer):
    """Enforce authoritative manual dimensions without replacing source topology."""

    stage_name = "dimension_calibration"

    def apply(self, context: DeformationContext) -> DeformationContext:
        m = context.measurements
        center_x = float(context.mesh("Bridge").vertices[:, 0].mean())

        half_frame_mm = m.frame_width * 0.5
        lens_outer_from_center_mm = m.bridge_width * 0.5 + m.lens_width
        side_margin_mm = half_frame_mm - lens_outer_from_center_mm
        if side_margin_mm <= 0.0:
            raise ValueError(
                "Inconsistent measurements: frame width must exceed "
                "2*lens_width + bridge_width."
            )

        before = self._measure(context)

        left_rim = context.mesh("LeftRim")
        right_rim = context.mesh("RightRim")
        left_lens = context.mesh("LeftLens")
        right_lens = context.mesh("RightLens")
        bridge = context.mesh("Bridge")
        left_temple = context.mesh("LeftTemple")
        right_temple = context.mesh("RightTemple")

        left_outer_before = float(left_rim.bounds[0, 0])
        right_outer_before = float(right_rim.bounds[1, 0])

        self._fit_bridge(bridge, center_x, m.bridge_width)

        self._fit_side(
            rim=left_rim,
            lens=left_lens,
            sign=-1.0,
            center_x=center_x,
            frame_width_mm=m.frame_width,
            bridge_width_mm=m.bridge_width,
            lens_width_mm=m.lens_width,
            lens_height_mm=m.lens_height,
            rim_thickness_mm=m.rim_thickness,
        )
        self._fit_side(
            rim=right_rim,
            lens=right_lens,
            sign=1.0,
            center_x=center_x,
            frame_width_mm=m.frame_width,
            bridge_width_mm=m.bridge_width,
            lens_width_mm=m.lens_width,
            lens_height_mm=m.lens_height,
            rim_thickness_mm=m.rim_thickness,
        )

        left_outer_after = float(left_rim.bounds[0, 0])
        right_outer_after = float(right_rim.bounds[1, 0])
        left_temple.apply_translation([left_outer_after - left_outer_before, 0.0, 0.0])
        right_temple.apply_translation([right_outer_after - right_outer_before, 0.0, 0.0])

        frame_vertices = np.vstack([left_rim.vertices, bridge.vertices, right_rim.vertices])
        self._fit_temple_length(left_temple, frame_vertices, m.temple_length)
        self._fit_temple_length(right_temple, frame_vertices, m.temple_length)

        # Frame is a runtime-only deformation/quality proxy. Keep its width
        # consistent with the exported component union without exporting it.
        frame = context.mesh("Frame")
        current_center = float((frame.bounds[0, 0] + frame.bounds[1, 0]) * 0.5)
        current_width = float(frame.extents[0])
        target_width = mm_to_m(m.frame_width)
        if current_width > 1e-9:
            vertices = frame.vertices.copy()
            vertices[:, 0] = center_x + (vertices[:, 0] - current_center) * (target_width / current_width)
            frame.vertices = vertices

        after = self._measure(context)
        errors = {
            key: round(after[key] - target, 6)
            for key, target in {
                "frame_width_mm": m.frame_width,
                "bridge_width_mm": m.bridge_width,
                "left_lens_width_mm": m.lens_width,
                "right_lens_width_mm": m.lens_width,
                "left_lens_height_mm": m.lens_height,
                "right_lens_height_mm": m.lens_height,
                "left_temple_length_mm": m.temple_length,
                "right_temple_length_mm": m.temple_length,
            }.items()
        }
        return self.update_context(
            context,
            applied=True,
            before=before,
            after=after,
            errors_mm=errors,
            side_margin_mm=round(side_margin_mm, 6),
        )

    @staticmethod
    def _fit_bridge(mesh, center_x: float, target_width_mm: float) -> None:
        vertices = mesh.vertices.copy()
        current_width = float(mesh.extents[0])
        if current_width <= 1e-9:
            raise ValueError("Bridge geometry has zero width")
        target_width = mm_to_m(target_width_mm)
        vertices[:, 0] = center_x + (vertices[:, 0] - float(vertices[:, 0].mean())) * (
            target_width / current_width
        )
        mesh.vertices = vertices

    @staticmethod
    def _map_outward(
        values: np.ndarray,
        *,
        current_inner: float,
        current_lens_outer: float,
        current_frame_outer: float,
        target_inner: float,
        target_lens_outer: float,
        target_frame_outer: float,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.empty_like(values)

        lens_den = max(current_lens_outer - current_inner, 1e-9)
        frame_den = max(current_frame_outer - current_lens_outer, 1e-9)

        inner_region = values <= current_lens_outer
        lens_t = (values[inner_region] - current_inner) / lens_den
        out[inner_region] = target_inner + lens_t * (target_lens_outer - target_inner)

        outer_region = ~inner_region
        outer_t = (values[outer_region] - current_lens_outer) / frame_den
        out[outer_region] = target_lens_outer + outer_t * (
            target_frame_outer - target_lens_outer
        )
        return out

    def _fit_side(
        self,
        *,
        rim,
        lens,
        sign: float,
        center_x: float,
        frame_width_mm: float,
        bridge_width_mm: float,
        lens_width_mm: float,
        lens_height_mm: float,
        rim_thickness_mm: float,
    ) -> None:
        rim_vertices = rim.vertices.copy()
        lens_vertices = lens.vertices.copy()

        rim_u = sign * (rim_vertices[:, 0] - center_x)
        lens_u = sign * (lens_vertices[:, 0] - center_x)

        current_rim_inner = float(np.min(rim_u))
        current_frame_outer = float(np.max(rim_u))
        current_rim_span = current_frame_outer - current_rim_inner
        if current_rim_span <= 1e-9:
            raise ValueError("Frame side has zero physical span")

        target_inner = mm_to_m(bridge_width_mm * 0.5)
        target_lens_outer = target_inner + mm_to_m(lens_width_mm)
        target_frame_outer = mm_to_m(frame_width_mm * 0.5)

        # Final calibration is authoritative. Earlier style stages may
        # temporarily place a lens outside the current rim envelope, so do not
        # derive the frame mapping from the pre-calibration lens extent.
        # Preserve every real rim vertex and map the source rim span directly
        # into the target bridge-to-frame span.
        rim_t = (rim_u - current_rim_inner) / current_rim_span
        rim_u_new = target_inner + rim_t * (target_frame_outer - target_inner)
        rim_vertices[:, 0] = center_x + sign * rim_u_new

        rim_center_y = float(np.mean(rim_vertices[:, 1]))
        current_rim_height = float(np.ptp(rim_vertices[:, 1]))
        target_rim_height = mm_to_m(lens_height_mm + 2.0 * rim_thickness_mm)
        if current_rim_height > 1e-9:
            rim_vertices[:, 1] = rim_center_y + (rim_vertices[:, 1] - rim_center_y) * (
                target_rim_height / current_rim_height
            )
        rim.vertices = rim_vertices

        lens_center_y = float(np.mean(lens_vertices[:, 1]))
        current_lens_width = float(np.ptp(lens_u))
        current_lens_height = float(np.ptp(lens_vertices[:, 1]))
        if current_lens_width <= 1e-9 or current_lens_height <= 1e-9:
            raise ValueError("Lens geometry has zero physical extent")

        target_center_u = target_inner + mm_to_m(lens_width_mm) * 0.5
        centered_u = lens_u - float(np.mean(lens_u))
        lens_u_new = target_center_u + centered_u * (
            mm_to_m(lens_width_mm) / current_lens_width
        )
        lens_vertices[:, 0] = center_x + sign * lens_u_new
        lens_vertices[:, 1] = lens_center_y + (lens_vertices[:, 1] - lens_center_y) * (
            mm_to_m(lens_height_mm) / current_lens_height
        )
        lens.vertices = lens_vertices

    @staticmethod
    def _fit_temple_length(mesh, frame_vertices: np.ndarray, target_length_mm: float) -> None:
        """Adjust hinge-to-tip distance while keeping the hinge fixed."""
        vertices = mesh.vertices.copy()
        if len(vertices) == 0:
            raise ValueError("Temple geometry is empty")
        tree = cKDTree(np.asarray(frame_vertices, dtype=np.float64))
        distances, _ = tree.query(vertices, k=1)
        hinge = vertices[int(np.argmin(distances))].copy()

        radial = vertices - hinge
        radial_distances = np.linalg.norm(radial, axis=1)
        tip_index = int(np.argmax(radial_distances))
        current_length = float(radial_distances[tip_index])
        if current_length <= 1e-9:
            raise ValueError("Temple hinge-to-tip length is zero")

        axis = radial[tip_index] / current_length
        target_length = mm_to_m(target_length_mm)
        delta = target_length - current_length
        axial = np.maximum(np.dot(radial, axis), 0.0)
        progress = np.clip(axial / current_length, 0.0, 1.0)
        vertices += (delta * progress)[:, None] * axis[None, :]
        mesh.vertices = vertices

    @staticmethod
    def _measure(context: DeformationContext) -> dict[str, float]:
        bridge = context.mesh("Bridge")
        left_rim = context.mesh("LeftRim")
        right_rim = context.mesh("RightRim")
        left_lens = context.mesh("LeftLens")
        right_lens = context.mesh("RightLens")

        frame_min = min(float(left_rim.bounds[0, 0]), float(bridge.bounds[0, 0]))
        frame_max = max(float(right_rim.bounds[1, 0]), float(bridge.bounds[1, 0]))

        frame_vertices = np.vstack([left_rim.vertices, bridge.vertices, right_rim.vertices])

        def temple_length_mm(name: str) -> float:
            temple = context.mesh(name)
            tree = cKDTree(frame_vertices)
            distances, _ = tree.query(temple.vertices, k=1)
            hinge = temple.vertices[int(np.argmin(distances))]
            return m_to_mm(float(np.max(np.linalg.norm(temple.vertices - hinge, axis=1))))

        return {
            "frame_width_mm": round(m_to_mm(frame_max - frame_min), 6),
            "bridge_width_mm": round(m_to_mm(bridge.extents[0]), 6),
            "left_lens_width_mm": round(m_to_mm(left_lens.extents[0]), 6),
            "right_lens_width_mm": round(m_to_mm(right_lens.extents[0]), 6),
            "left_lens_height_mm": round(m_to_mm(left_lens.extents[1]), 6),
            "right_lens_height_mm": round(m_to_mm(right_lens.extents[1]), 6),
            "left_temple_length_mm": round(temple_length_mm("LeftTemple"), 6),
            "right_temple_length_mm": round(temple_length_mm("RightTemple"), 6),
        }
