"""Topology-aware rim deformation driven by lens contours and measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.models import LensContour


@dataclass(frozen=True)
class RimSideResult:
    """Debug summary for one rim deformation pass."""

    side: str
    width_before: float
    width_after: float
    height_before: float
    height_after: float
    contour_points: int


class RimDeformer(BaseDeformer):
    """Reshape left and right rim loops while preserving topology and thickness."""

    stage_name = "rim_deformation"

    def __init__(self, contour_strength: float = 0.7):
        self.contour_strength = float(np.clip(contour_strength, 0.0, 1.0))

    def apply(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> DeformationContext:
        contour = lens_contour or LensContour()
        results: list[RimSideResult] = []

        for side in ("left", "right"):
            rim_group = f"{side}_rim"
            rim_parts = context.descriptor.vertex_groups.get(rim_group, [])
            if not rim_parts:
                continue

            rim_mesh = context.mesh(rim_parts[0])
            frame_meshes = context.vertex_group_meshes("frame")
            if not frame_meshes:
                continue
            frame_mesh = frame_meshes[0]

            before = rim_mesh.vertices.copy()
            target_contour = self._target_contour(context, side, contour)
            self._deform_rim_mesh(rim_mesh, side, target_contour, context.measurements.rim_thickness)
            self._propagate_to_frame(frame_mesh, before, rim_mesh.vertices)

            bounds_before = np.ptp(before[:, :2], axis=0)
            bounds_after = np.ptp(rim_mesh.vertices[:, :2], axis=0)
            results.append(
                RimSideResult(
                    side=side,
                    width_before=round(float(bounds_before[0]), 3),
                    width_after=round(float(bounds_after[0]), 3),
                    height_before=round(float(bounds_before[1]), 3),
                    height_after=round(float(bounds_after[1]), 3),
                    contour_points=len(target_contour),
                )
            )

        return self.update_context(
            context,
            applied=bool(results),
            sides=[result.__dict__ for result in results],
            contour_strength=self.contour_strength,
        )

    def _deform_rim_mesh(
        self,
        rim_mesh,
        side: str,
        target_contour: np.ndarray,
        target_thickness: float,
    ) -> None:
        vertices = rim_mesh.vertices.copy()
        front_indices, back_indices = self._split_surface_indices(vertices)
        if len(front_indices) == 0 or len(back_indices) == 0:
            return

        front_vertices = vertices[front_indices]
        back_vertices = vertices[back_indices]

        front_order = self._loop_order(front_vertices[:, :2])
        back_order = self._loop_order(back_vertices[:, :2])
        ordered_front_idx = front_indices[front_order]
        ordered_back_idx = back_indices[back_order]

        template_front = vertices[ordered_front_idx][:, :2]
        resampled_target = self._resample_contour(target_contour, len(ordered_front_idx))
        resampled_target = self._align_contour_to_template(template_front, resampled_target)
        blended_target = self._blend_template_and_target(template_front, resampled_target)

        front_centroid = template_front.mean(axis=0)
        target_centroid = blended_target.mean(axis=0)
        front_radial = np.linalg.norm(template_front - front_centroid, axis=1)
        target_radial = np.linalg.norm(blended_target - target_centroid, axis=1)
        radial_ratio = np.divide(target_radial, np.maximum(front_radial, 1e-6))

        vertices[ordered_front_idx, 0:2] = blended_target
        thickness = self._target_depth(vertices, ordered_front_idx, ordered_back_idx, target_thickness)

        back_template = vertices[ordered_back_idx][:, :2]
        back_centroid = back_template.mean(axis=0)
        centered_back = back_template - back_centroid
        scaled_back = centered_back * radial_ratio[:, None]
        vertices[ordered_back_idx, 0:2] = target_centroid + scaled_back

        front_z = float(np.max(vertices[ordered_front_idx, 2]))
        vertices[ordered_front_idx, 2] = front_z
        vertices[ordered_back_idx, 2] = front_z - thickness

        rim_mesh.vertices = vertices

    def _target_contour(
        self,
        context: DeformationContext,
        side: str,
        lens_contour: LensContour,
    ) -> np.ndarray:
        raw_points = lens_contour.left if side == "left" else lens_contour.right
        desired = self._contour_points_to_mm(raw_points, side, context)
        if desired is None:
            desired = self._fallback_contour(context, side)
        return desired

    def _contour_points_to_mm(
        self,
        points: list[list[float]],
        side: str,
        context: DeformationContext,
    ) -> np.ndarray | None:
        if len(points) < 4:
            return None

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None

        if side == "left":
            local_x = arr[:, 0] / 0.5
        else:
            local_x = (arr[:, 0] - 0.5) / 0.5
        local_y = arr[:, 1]

        local_x = np.clip(local_x, 0.0, 1.0)
        local_y = np.clip(local_y, 0.0, 1.0)

        measured = context.measurements
        center_x = (-1.0 if side == "left" else 1.0) * (measured.bridge_width * 0.5 + measured.lens_width * 0.5)
        center_y = 0.0
        width = measured.lens_width
        height = measured.lens_height

        x_mm = center_x + (local_x - 0.5) * width
        y_mm = center_y + (0.5 - local_y) * height

        contour = np.column_stack([x_mm, y_mm])
        return self._normalize_contour_orientation(contour, side)

    def _fallback_contour(self, context: DeformationContext, side: str) -> np.ndarray:
        plane = context.descriptor.lens_planes[side]
        width = context.measurements.lens_width
        height = context.measurements.lens_height
        cx, cy = plane.origin[0], plane.origin[1]
        base = np.array(
            [
                [-0.5, 0.5],
                [-0.72, 0.22],
                [-0.72, -0.22],
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.72, -0.22],
                [0.72, 0.22],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        contour = np.column_stack([cx + base[:, 0] * width, cy + base[:, 1] * height])
        return self._normalize_contour_orientation(contour, side)

    @staticmethod
    def _normalize_contour_orientation(contour: np.ndarray, side: str) -> np.ndarray:
        center = contour.mean(axis=0)
        angles = np.arctan2(contour[:, 1] - center[1], contour[:, 0] - center[0])
        order = np.argsort(-angles)
        ordered = contour[order]
        start = np.argmin(ordered[:, 0] if side == "left" else -ordered[:, 0])
        return np.roll(ordered, -start, axis=0)

    @staticmethod
    def _split_surface_indices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_values = vertices[:, 2]
        front_z = float(np.max(z_values))
        back_z = float(np.min(z_values))
        front_indices = np.where(np.isclose(z_values, front_z))[0]
        back_indices = np.where(np.isclose(z_values, back_z))[0]
        return front_indices, back_indices

    @staticmethod
    def _loop_order(points_2d: np.ndarray) -> np.ndarray:
        center = points_2d.mean(axis=0)
        angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
        return np.argsort(-angles)

    def _resample_contour(self, contour: np.ndarray, count: int) -> np.ndarray:
        if len(contour) == count:
            return contour.copy()

        closed = np.vstack([contour, contour[0]])
        distances = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        if cumulative[-1] <= 1e-6:
            return np.repeat(contour[:1], count, axis=0)

        t = cumulative / cumulative[-1]
        k = min(3, len(contour) - 1)
        spline_x = interpolate.make_interp_spline(t, closed[:, 0], k=k)
        spline_y = interpolate.make_interp_spline(t, closed[:, 1], k=k)
        sample_t = np.linspace(0.0, 1.0, count + 1)[:-1]
        return np.column_stack([spline_x(sample_t), spline_y(sample_t)])

    def _blend_template_and_target(
        self,
        template_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        template_center = template_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        template_shape = template_points - template_center
        target_shape = target_points - target_center
        blended_shape = template_shape * (1.0 - self.contour_strength) + target_shape * self.contour_strength
        return target_center + blended_shape

    @staticmethod
    def _align_contour_to_template(template_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Find the best cyclic shift + optional reversal using FFT cross-correlation (O(n log n))."""
        n = len(template_points)
        if n == 0:
            return target_points

        best = target_points
        best_score = float("inf")

        for candidate in (target_points, target_points[::-1]):
            # Use x-coordinate cross-correlation to find best cyclic shift cheaply
            t_x = template_points[:, 0] - template_points[:, 0].mean()
            c_x = candidate[:, 0] - candidate[:, 0].mean()
            if len(t_x) != len(c_x):
                # Fallback to direct if sizes mismatch
                score = float(np.sum(np.linalg.norm(template_points - candidate, axis=1)))
                if score < best_score:
                    best_score = score
                    best = candidate
                continue

            # FFT cross-correlation on x-coords to find best shift cheaply
            corr = np.real(np.fft.ifft(np.fft.fft(t_x) * np.conj(np.fft.fft(c_x))))
            best_shift = int(np.argmax(corr))

            # Check best_shift and its neighbours (±1) for robustness
            for shift in (best_shift - 1, best_shift, best_shift + 1):
                rolled = np.roll(candidate, shift % n, axis=0)
                score = float(np.sum(np.linalg.norm(template_points - rolled, axis=1)))
                if score < best_score:
                    best_score = score
                    best = rolled

        return best

    @staticmethod
    def _target_depth(
        vertices: np.ndarray,
        front_indices: np.ndarray,
        back_indices: np.ndarray,
        target_thickness: float,
    ) -> float:
        existing = float(np.mean(vertices[front_indices, 2] - vertices[back_indices, 2]))
        return float(np.clip(target_thickness, 0.8, 6.0)) if np.isfinite(target_thickness) else existing

    @staticmethod
    def _propagate_to_frame(frame_mesh, original_rim_vertices: np.ndarray, deformed_rim_vertices: np.ndarray) -> None:
        """Propagate rim deformation into frame mesh — vectorised with spatial index."""
        from scipy.spatial import cKDTree

        frame_vertices = frame_mesh.vertices.copy()
        deltas = deformed_rim_vertices - original_rim_vertices  # (R, 3)

        tree = cKDTree(frame_vertices)

        # --- Exact matches (within 1e-5) ---
        exact_indices = tree.query_ball_point(original_rim_vertices, r=1e-5)
        for r_idx, f_indices in enumerate(exact_indices):
            if f_indices:
                frame_vertices[f_indices] += deltas[r_idx]

        # --- Soft falloff within radius 12 mm ---
        # Only query rim verts that had no exact match
        no_exact_mask = np.array([len(fi) == 0 for fi in exact_indices])
        if not no_exact_mask.any():
            frame_mesh.vertices = frame_vertices
            return

        soft_rim = original_rim_vertices[no_exact_mask]
        soft_deltas = deltas[no_exact_mask]

        # Query neighbours within 12 mm
        neighbours = tree.query_ball_point(soft_rim, r=12.0)
        weighted = np.zeros_like(frame_vertices)
        for r_idx, f_indices in enumerate(neighbours):
            if not f_indices:
                continue
            f_idx = np.array(f_indices, dtype=np.int32)
            dist = np.linalg.norm(frame_vertices[f_idx] - soft_rim[r_idx], axis=1)
            influence = BaseDeformer.apply_falloff(np.clip(1.0 - dist / 12.0, 0.0, 1.0))
            weighted[f_idx] += soft_deltas[r_idx] * influence[:, None] * 0.2

        frame_vertices += weighted
        frame_mesh.vertices = frame_vertices
