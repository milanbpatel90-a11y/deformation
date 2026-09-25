"""Rim-driven lens deformation that preserves topology, UVs, thickness, and curvature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.geometry_units import m_to_mm, mm_to_m


@dataclass(frozen=True)
class LensSelection:
    """Lens-local meshes and source geometry for one side."""

    side: str
    lens_part: str
    rim_part: str
    lens_before: np.ndarray
    rim_before: np.ndarray


class LensDeformer(BaseDeformer):
    """Fit lenses to the current rim boundary while preserving optical properties."""

    stage_name = "lens_deformation"

    def __init__(self, rim_inset: float = 2.0):
        self.rim_inset = float(max(0.4, rim_inset))

    def apply(self, context: DeformationContext) -> DeformationContext:
        results: list[dict[str, Any]] = []
        for side in ("left", "right"):
            selection = self._collect_lenses(context, side)
            rim_boundary = self._extract_rim_boundary(context, selection)
            target_boundary = self._generate_target_boundary(context, selection, rim_boundary)
            self._fit_lens_to_rim(context, selection, target_boundary)
            self._preserve_uvs(context, selection)
            self._restore_source_normal_profile(context, selection)
            results.append(self._validate(context, selection, target_boundary))
        return self.update_context(context, applied=True, sides=results)

    def _collect_lenses(self, context: DeformationContext, side: str) -> LensSelection:
        lens_part = context.descriptor.vertex_groups[f"{side}_lens"][0]
        rim_part = context.descriptor.vertex_groups[f"{side}_rim"][0]
        return LensSelection(
            side=side,
            lens_part=lens_part,
            rim_part=rim_part,
            lens_before=context.mesh(lens_part).vertices.copy(),
            rim_before=context.mesh(rim_part).vertices.copy(),
        )

    def _extract_rim_boundary(self, context: DeformationContext, selection: LensSelection) -> np.ndarray:
        rim_mesh = context.mesh(selection.rim_part)
        origin, basis = self._lens_basis(selection)
        rim_local = self._to_local(np.asarray(rim_mesh.vertices, dtype=np.float64), origin, basis)
        return self._convex_boundary(rim_local[:, :2])

    def _generate_target_boundary(
        self,
        context: DeformationContext,
        selection: LensSelection,
        rim_boundary: np.ndarray,
    ) -> np.ndarray:
        center = rim_boundary.mean(axis=0)
        radial = rim_boundary - center
        lengths = np.linalg.norm(radial, axis=1, keepdims=True)
        safe_lengths = np.maximum(lengths, 1e-9)
        rim_inset_m = mm_to_m(self.rim_inset)
        minimum_edge_m = mm_to_m(0.25)
        minimum_inset_m = mm_to_m(0.15)
        inset = np.minimum(rim_inset_m, lengths - minimum_edge_m)
        inset = np.maximum(inset, minimum_inset_m)
        target = center + radial * ((safe_lengths - inset) / safe_lengths)
        return target

    def _fit_lens_to_rim(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = np.asarray(lens_mesh.vertices, dtype=np.float64).copy()
        origin, basis = self._lens_basis(selection)
        local = self._to_local(vertices, origin, basis)
        front_idx = self._surface_indices(local, front=True)
        back_idx = self._surface_indices(local, front=False)

        front_boundary = local[front_idx, :2]
        front_order = self._loop_order(front_boundary)
        back_boundary = local[back_idx, :2]
        back_order = self._loop_order(back_boundary)

        ordered_front_idx = front_idx[front_order]
        ordered_back_idx = back_idx[back_order]
        template_front = local[ordered_front_idx, :2]
        template_back = local[ordered_back_idx, :2]

        resampled_target = self._resample_boundary(target_boundary, len(ordered_front_idx))
        resampled_target = self._align_boundary(template_front, resampled_target)
        local[ordered_front_idx, :2] = resampled_target

        # Fit the back surface independently because production lenses may
        # have a different vertex count on front and back. Reusing the front
        # scale array would corrupt or fail on asymmetric tessellation.
        back_target = self._resample_boundary(target_boundary, len(ordered_back_idx))
        back_target = self._align_boundary(template_back, back_target)
        local[ordered_back_idx, :2] = back_target

        lens_mesh.vertices = self._from_local(local, origin, basis)

    def _preserve_uvs(self, context: DeformationContext, selection: LensSelection) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        if hasattr(lens_mesh.visual, "uv") and lens_mesh.visual.uv is not None:
            lens_mesh.visual.uv = lens_mesh.visual.uv.copy()

    def _restore_source_normal_profile(
        self,
        context: DeformationContext,
        selection: LensSelection,
    ) -> None:
        """Preserve source thickness and curvature exactly along the lens normal.

        The profile fit modifies only local X/Y. Reprojecting or flattening the
        front/back surfaces after that fit changes physical thickness on curved
        production lenses. Restore each source vertex's local normal coordinate
        one-for-one so topology, optical thickness, and curvature are preserved
        while the in-plane silhouette changes.
        """
        lens_mesh = context.mesh(selection.lens_part)
        vertices = np.asarray(lens_mesh.vertices, dtype=np.float64).copy()
        origin, basis = self._lens_basis(selection)
        local = self._to_local(vertices, origin, basis)
        original_local = self._to_local(selection.lens_before, origin, basis)

        if len(local) != len(original_local):
            raise ValueError(
                f"Lens topology changed for {selection.lens_part}: "
                f"{len(original_local)} -> {len(local)} vertices"
            )

        local[:, 2] = original_local[:, 2]
        lens_mesh.vertices = self._from_local(local, origin, basis)

    # Backward-compatible private helpers used by older callers/tests.
    def _preserve_thickness(self, context: DeformationContext, selection: LensSelection) -> None:
        self._restore_source_normal_profile(context, selection)

    def _preserve_curvature(self, context: DeformationContext, selection: LensSelection) -> None:
        self._restore_source_normal_profile(context, selection)

    def _validate(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> dict[str, Any]:
        lens_mesh = context.mesh(selection.lens_part)
        rim_mesh = context.mesh(selection.rim_part)
        lens_vertices = np.asarray(lens_mesh.vertices, dtype=np.float64)
        rim_vertices = np.asarray(rim_mesh.vertices, dtype=np.float64)
        origin, basis = self._lens_basis(selection)
        lens_local = self._to_local(lens_vertices, origin, basis)
        rim_local = self._to_local(rim_vertices, origin, basis)

        front_idx = self._surface_indices(lens_local, front=True)
        back_idx = self._surface_indices(lens_local, front=False)
        lens_boundary = lens_local[front_idx, :2]
        lens_boundary = lens_boundary[self._loop_order(lens_boundary)]
        fitted_target = self._resample_boundary(target_boundary, len(lens_boundary))
        fitted_target = self._align_boundary(lens_boundary, fitted_target)
        fit_error = m_to_mm(
            float(np.mean(np.linalg.norm(lens_boundary - fitted_target, axis=1)))
        )

        rim_boundary = self._convex_boundary(rim_local[:, :2])
        lens_polygon_boundary = self._convex_boundary(lens_boundary)
        rim_polygon = Polygon(rim_boundary).buffer(0)
        lens_polygon = Polygon(lens_polygon_boundary).buffer(0)
        inside_rim = bool(
            not rim_polygon.is_empty
            and not lens_polygon.is_empty
            and rim_polygon.covers(lens_polygon)
        )

        thickness = m_to_mm(
            float(
                np.mean(lens_local[front_idx, 2])
                - np.mean(lens_local[back_idx, 2])
            )
        )
        original_local = self._to_local(selection.lens_before, origin, basis)
        curvature_delta = float(
            np.std(lens_local[:, 2]) - np.std(original_local[:, 2])
        )

        uv_count = 0
        if hasattr(lens_mesh.visual, "uv") and lens_mesh.visual.uv is not None:
            uv_count = int(len(lens_mesh.visual.uv))
        return self.validate_constraints(
            {
                "side": selection.side,
                "vertex_count": int(len(lens_vertices)),
                "uv_count": uv_count,
                "fit_error_mm": round(fit_error, 6),
                "thickness_mm": round(thickness, 6),
                "curvature_delta": round(curvature_delta, 9),
                "inside_rim": inside_rim,
                "valid": bool(fit_error < 2.5 and thickness > 0.0 and inside_rim),
            }
        )

    @staticmethod
    def _surface_indices(local_vertices: np.ndarray, front: bool) -> np.ndarray:
        normal_values = local_vertices[:, 2]
        target = float(np.max(normal_values) if front else np.min(normal_values))
        span = max(float(np.ptp(normal_values)), 1e-9)
        tolerance = max(span * 0.08, 1e-8)
        indices = np.where(np.abs(normal_values - target) <= tolerance)[0]
        if len(indices) < 3:
            count = min(max(3, len(local_vertices) // 12), len(local_vertices))
            order = np.argsort(normal_values)
            indices = order[-count:] if front else order[:count]
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _lens_basis(selection: LensSelection) -> tuple[np.ndarray, np.ndarray]:
        """Return stable world→lens-local PCA basis from source lens geometry."""
        vertices = np.asarray(selection.lens_before, dtype=np.float64)
        origin = vertices.mean(axis=0)
        centered = vertices - origin
        covariance = centered.T @ centered / max(len(centered), 1)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        plane_u = vectors[:, order[0]]
        plane_v = vectors[:, order[1]]
        normal = vectors[:, order[2]]

        # Keep a right-handed orthonormal basis and stable normal orientation.
        plane_u = plane_u / max(np.linalg.norm(plane_u), 1e-12)
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        plane_v = np.cross(normal, plane_u)
        plane_v = plane_v / max(np.linalg.norm(plane_v), 1e-12)
        normal = np.cross(plane_u, plane_v)
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        basis = np.column_stack([plane_u, plane_v, normal])
        return origin, basis

    @staticmethod
    def _to_local(vertices: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return (np.asarray(vertices, dtype=np.float64) - origin) @ basis

    @staticmethod
    def _from_local(local: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.asarray(local, dtype=np.float64) @ basis.T + origin

    @staticmethod
    def _convex_boundary(points_2d: np.ndarray) -> np.ndarray:
        points = np.unique(np.asarray(points_2d, dtype=np.float64), axis=0)
        if len(points) < 3:
            raise ValueError("Projected rim/lens boundary has fewer than 3 unique points")
        hull = ConvexHull(points)
        return points[hull.vertices]

    @staticmethod
    def _loop_order(points_2d: np.ndarray) -> np.ndarray:
        center = points_2d.mean(axis=0)
        angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
        return np.argsort(-angles)

    def _resample_boundary(self, boundary: np.ndarray, count: int) -> np.ndarray:
        if len(boundary) == count:
            return boundary.copy()
        closed = np.vstack([boundary, boundary[0]])
        distances = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        if cumulative[-1] <= 1e-6:
            return np.repeat(boundary[:1], count, axis=0)
        t = cumulative / cumulative[-1]
        k = min(3, len(boundary) - 1)
        spline_x = interpolate.make_interp_spline(t, closed[:, 0], k=k)
        spline_y = interpolate.make_interp_spline(t, closed[:, 1], k=k)
        sample_t = np.linspace(0.0, 1.0, count + 1)[:-1]
        return np.column_stack([spline_x(sample_t), spline_y(sample_t)])

    @staticmethod
    def _align_boundary(template_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Find best cyclic shift + optional reversal using FFT cross-correlation (O(n log n))."""
        n = len(template_points)
        if n == 0:
            return target_points

        best = target_points
        best_score = float("inf")

        for candidate in (target_points, target_points[::-1]):
            if len(candidate) != n:
                score = float(np.sum(np.linalg.norm(template_points - candidate, axis=1)))
                if score < best_score:
                    best_score = score
                    best = candidate
                continue

            t_x = template_points[:, 0] - template_points[:, 0].mean()
            c_x = candidate[:, 0] - candidate[:, 0].mean()
            corr = np.real(np.fft.ifft(np.fft.fft(t_x) * np.conj(np.fft.fft(c_x))))
            best_shift = int(np.argmax(corr))

            for shift in (best_shift - 1, best_shift, best_shift + 1):
                rolled = np.roll(candidate, shift % n, axis=0)
                score = float(np.sum(np.linalg.norm(template_points - rolled, axis=1)))
                if score < best_score:
                    best_score = score
                    best = rolled

        return best
