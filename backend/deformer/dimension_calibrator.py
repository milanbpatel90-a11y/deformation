"""Final millimetre calibration for deformed eyewear components.

The style stages are allowed to reshape vertices, but the supplied physical
measurements are authoritative. This stage bounds every transformation, keeps
left/right placement symmetric, and reports measured errors from real geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


class DimensionCalibrator(BaseDeformer):
    """Enforce requested dimensions without changing mesh topology."""

    stage_name = "dimension_calibration"

    def apply(self, context: DeformationContext) -> DeformationContext:
        m = context.measurements
        side_margin = m.frame_width - (2.0 * m.lens_width + m.bridge_width)
        if side_margin <= 0.0:
            raise ValueError(
                "Inconsistent measurements: frame width must exceed "
                "2*lens_width + bridge_width."
            )

        required = (
            "Frame", "Bridge", "LeftRim", "RightRim",
            "LeftLens", "RightLens", "LeftTemple", "RightTemple",
        )
        missing = [name for name in required if name not in context.meshes]
        if missing:
            raise ValueError("Cannot calibrate missing components: " + ", ".join(missing))

        before = self._measure(context)
        center_x = self._symmetry_center(context)

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
            rim=left_rim, lens=left_lens, sign=-1.0, center_x=center_x,
            frame_width=m.frame_width, bridge_width=m.bridge_width,
            lens_width=m.lens_width, lens_height=m.lens_height,
            rim_thickness=m.rim_thickness,
        )
        self._fit_side(
            rim=right_rim, lens=right_lens, sign=1.0, center_x=center_x,
            frame_width=m.frame_width, bridge_width=m.bridge_width,
            lens_width=m.lens_width, lens_height=m.lens_height,
            rim_thickness=m.rim_thickness,
        )
        self._fit_frame(context, center_x)

        # Keep the temples attached when the front frame width changes.
        left_temple.apply_translation([
            float(left_rim.bounds[0, 0]) - left_outer_before, 0.0, 0.0
        ])
        right_temple.apply_translation([
            float(right_rim.bounds[1, 0]) - right_outer_before, 0.0, 0.0
        ])

        frame_vertices = np.vstack([
            left_rim.vertices, bridge.vertices, right_rim.vertices,
            context.mesh("Frame").vertices,
        ])
        self._fit_temple_length(left_temple, frame_vertices, m.temple_length)
        self._fit_temple_length(right_temple, frame_vertices, m.temple_length)

        after = self._measure(context)
        targets = {
            "frame_width_mm": m.frame_width,
            "bridge_width_mm": m.bridge_width,
            "left_lens_width_mm": m.lens_width,
            "right_lens_width_mm": m.lens_width,
            "left_lens_height_mm": m.lens_height,
            "right_lens_height_mm": m.lens_height,
            "left_temple_length_mm": m.temple_length,
            "right_temple_length_mm": m.temple_length,
        }
        errors = {key: round(after[key] - target, 6) for key, target in targets.items()}
        return self.update_context(
            context,
            applied=True,
            before=before,
            after=after,
            targets_mm=targets,
            errors_mm=errors,
            max_abs_error_mm=round(max(abs(value) for value in errors.values()), 6),
            side_margin_mm=round(side_margin, 6),
        )

    @staticmethod
    def _symmetry_center(context: DeformationContext) -> float:
        plane = context.descriptor.raw.get("symmetry_plane", {})
        point = plane.get("point", [0.0, 0.0, 0.0]) if isinstance(plane, dict) else [0.0, 0.0, 0.0]
        try:
            return float(point[0])
        except (TypeError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def _fit_bridge(mesh, center_x: float, target_width: float) -> None:
        vertices = mesh.vertices.copy()
        current_width = float(np.ptp(vertices[:, 0]))
        if current_width <= 1e-9:
            raise ValueError("Bridge geometry has zero width")
        current_center = float((vertices[:, 0].min() + vertices[:, 0].max()) * 0.5)
        vertices[:, 0] = center_x + (vertices[:, 0] - current_center) * (target_width / current_width)
        mesh.vertices = vertices

    @staticmethod
    def _fit_side(
        *, rim, lens, sign: float, center_x: float, frame_width: float,
        bridge_width: float, lens_width: float, lens_height: float,
        rim_thickness: float,
    ) -> None:
        rim_vertices = rim.vertices.copy()
        lens_vertices = lens.vertices.copy()
        rim_u = sign * (rim_vertices[:, 0] - center_x)
        lens_u = sign * (lens_vertices[:, 0] - center_x)

        current_rim_inner = float(rim_u.min())
        current_rim_outer = float(rim_u.max())
        current_rim_span = current_rim_outer - current_rim_inner
        if current_rim_span <= 1e-9:
            raise ValueError("Frame side has zero physical span")

        target_inner = bridge_width * 0.5
        target_outer = frame_width * 0.5
        rim_t = (rim_u - current_rim_inner) / current_rim_span
        rim_u_new = target_inner + rim_t * (target_outer - target_inner)
        rim_vertices[:, 0] = center_x + sign * rim_u_new

        current_lens_width = float(np.ptp(lens_u))
        current_lens_height = float(np.ptp(lens_vertices[:, 1]))
        if current_lens_width <= 1e-9 or current_lens_height <= 1e-9:
            raise ValueError("Lens geometry has zero physical extent")

        target_lens_center_u = target_inner + lens_width * 0.5
        current_lens_center_u = float((lens_u.min() + lens_u.max()) * 0.5)
        lens_u_new = target_lens_center_u + (lens_u - current_lens_center_u) * (
            lens_width / current_lens_width
        )
        lens_center_y = float((lens_vertices[:, 1].min() + lens_vertices[:, 1].max()) * 0.5)
        lens_vertices[:, 0] = center_x + sign * lens_u_new
        lens_vertices[:, 1] = lens_center_y + (lens_vertices[:, 1] - lens_center_y) * (
            lens_height / current_lens_height
        )

        # Center the rim on the lens and leave the requested physical rim
        # thickness around it. This prevents a lens from extending past its rim.
        current_rim_height = float(np.ptp(rim_vertices[:, 1]))
        if current_rim_height <= 1e-9:
            raise ValueError("Rim geometry has zero physical height")
        rim_center_y = float((rim_vertices[:, 1].min() + rim_vertices[:, 1].max()) * 0.5)
        target_rim_height = lens_height + 2.0 * rim_thickness
        rim_vertices[:, 1] = lens_center_y + (rim_vertices[:, 1] - rim_center_y) * (
            target_rim_height / current_rim_height
        )

        rim.vertices = rim_vertices
        lens.vertices = lens_vertices

    @staticmethod
    def _fit_frame(context: DeformationContext, center_x: float) -> None:
        """Restore the brow/frame part from its source shape and fit the new front."""
        frame = context.mesh("Frame")
        source_frame = context.source_meshes.get("Frame")
        source_left = context.source_meshes.get("LeftLens")
        source_right = context.source_meshes.get("RightLens")
        if source_frame is None or source_left is None or source_right is None:
            return

        vertices = source_frame.vertices.copy()
        source_lenses = (source_left, source_right)
        source_lens_height = float(np.mean([mesh.extents[1] for mesh in source_lenses]))
        if source_lens_height <= 1e-9:
            return
        scale_y = context.measurements.lens_height / source_lens_height

        source_lens_center_y = float(np.mean([
            (mesh.bounds[0, 1] + mesh.bounds[1, 1]) * 0.5 for mesh in source_lenses
        ]))
        target_lens_center_y = float(np.mean([
            (context.mesh(name).bounds[0, 1] + context.mesh(name).bounds[1, 1]) * 0.5
            for name in ("LeftLens", "RightLens")
        ]))
        source_frame_center_y = float((source_frame.bounds[0, 1] + source_frame.bounds[1, 1]) * 0.5)
        target_frame_center_y = target_lens_center_y + (
            source_frame_center_y - source_lens_center_y
        ) * scale_y
        source_frame_center_x = float((source_frame.bounds[0, 0] + source_frame.bounds[1, 0]) * 0.5)

        vertices[:, 0] = center_x + (vertices[:, 0] - source_frame_center_x) * (
            context.measurements.frame_width / max(float(source_frame.extents[0]), 1e-9)
        )
        vertices[:, 1] = target_frame_center_y + (vertices[:, 1] - source_frame_center_y) * scale_y

        source_lens_center_z = float(np.mean([
            (mesh.bounds[0, 2] + mesh.bounds[1, 2]) * 0.5 for mesh in source_lenses
        ]))
        target_lens_center_z = float(np.mean([
            (context.mesh(name).bounds[0, 2] + context.mesh(name).bounds[1, 2]) * 0.5
            for name in ("LeftLens", "RightLens")
        ]))
        vertices[:, 2] += target_lens_center_z - source_lens_center_z
        frame.vertices = vertices

    @staticmethod
    def _fit_temple_length(mesh, frame_vertices: np.ndarray, target_length: float) -> None:
        vertices = mesh.vertices.copy()
        if len(vertices) == 0:
            raise ValueError("Temple geometry is empty")
        distances, _ = cKDTree(np.asarray(frame_vertices, dtype=np.float64)).query(vertices, k=1)
        hinge = vertices[int(np.argmin(distances))].copy()
        radial = vertices - hinge
        radial_distances = np.linalg.norm(radial, axis=1)
        tip_index = int(np.argmax(radial_distances))
        current_length = float(radial_distances[tip_index])
        if current_length <= 1e-9:
            raise ValueError("Temple hinge-to-tip length is zero")

        axis = radial[tip_index] / current_length
        axial = np.maximum(np.dot(radial, axis), 0.0)
        progress = np.clip(axial / current_length, 0.0, 1.0)
        vertices += ((target_length - current_length) * progress)[:, None] * axis[None, :]
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

        def temple_length(name: str) -> float:
            temple = context.mesh(name)
            distances, _ = cKDTree(frame_vertices).query(temple.vertices, k=1)
            hinge = temple.vertices[int(np.argmin(distances))]
            return float(np.max(np.linalg.norm(temple.vertices - hinge, axis=1)))

        return {
            "frame_width_mm": round(frame_max - frame_min, 6),
            "bridge_width_mm": round(float(bridge.extents[0]), 6),
            "left_lens_width_mm": round(float(left_lens.extents[0]), 6),
            "right_lens_width_mm": round(float(right_lens.extents[0]), 6),
            "left_lens_height_mm": round(float(left_lens.extents[1]), 6),
            "right_lens_height_mm": round(float(right_lens.extents[1]), 6),
            "left_temple_length_mm": round(temple_length("LeftTemple"), 6),
            "right_temple_length_mm": round(temple_length("RightTemple"), 6),
        }
