"""Rim-driven lens deformation that preserves topology, UVs, thickness, and curvature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull, QhullError

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


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
            fit_scale = self._fit_lens_to_rim(context, selection, target_boundary)
            self._preserve_uvs(context, selection)
            self._preserve_thickness(context, selection, fit_scale)
            self._preserve_curvature(context, selection, fit_scale)
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
        rim_vertices = rim_mesh.vertices.copy()
        front_idx = self._surface_indices(rim_vertices, front=True)
        boundary = self._hull_boundary(rim_vertices[front_idx][:, :2])
        order = self._loop_order(boundary)
        return boundary[order]

    def _generate_target_boundary(
        self,
        context: DeformationContext,
        selection: LensSelection,
        rim_boundary: np.ndarray,
    ) -> np.ndarray:
        center = rim_boundary.mean(axis=0)
        radial = rim_boundary - center
        lengths = np.linalg.norm(radial, axis=1, keepdims=True)
        safe_lengths = np.maximum(lengths, 1e-6)
        inset = np.minimum(self.rim_inset, lengths - 0.25)
        inset = np.maximum(inset, 0.15)
        target = center + radial * ((safe_lengths - inset) / safe_lengths)
        return target

    def _fit_lens_to_rim(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> float:
        # NOTE on design: front_idx/back_idx (from _surface_indices) are *every*
        # vertex on the front/back surface, not just its outer edge. That is the
        # same set for a lens authored as a thin boundary loop (the engine's
        # original assumption), but for a lens authored as a real solid/filled
        # surface -- geometric_metal.glb's Cube.002_glass_mat_0, and every
        # Rhino-scan-derived lens built for the newer templates (cat_eye_gold,
        # oversized_clear, rectangle_metal, round_wire_metal, slim_rectangle_steel)
        # -- front_idx also includes hundreds/thousands of interior vertices.
        # Snapping every front vertex directly onto the resampled target boundary
        # (the previous behaviour) collapsed a solid lens face onto a 1D ring.
        #
        # Fix: derive a thin-plate-spline warp from (lens hull -> target
        # boundary) and apply it to every front vertex. At the hull itself this
        # is an exact interpolation (hull vertices land exactly on their target
        # points, so downstream fit-error validation sees ~0), while interior
        # vertices are smoothly carried along -- a true conforming fit rather
        # than a single rigid scale, which could only match the target's overall
        # size, not its actual (generally non-circular, non-similar) outline
        # shape. The same TPS warp also drives the back surface so front and
        # back stay a single coherent lens shape, offset only by thickness.
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)

        front_xy = vertices[front_idx][:, :2]
        back_xy = vertices[back_idx][:, :2]

        hull_xy = self._hull_boundary(front_xy)
        hull_order = self._loop_order(hull_xy)
        ordered_hull = hull_xy[hull_order]

        resampled_target = self._resample_boundary(target_boundary, len(ordered_hull))
        resampled_target = self._align_boundary(ordered_hull, resampled_target)

        hull_center = ordered_hull.mean(axis=0)
        target_center = resampled_target.mean(axis=0)
        hull_radius = float(np.linalg.norm(ordered_hull - hull_center, axis=1).mean())
        target_radius = float(np.linalg.norm(resampled_target - target_center, axis=1).mean())
        scale = target_radius / max(hull_radius, 1e-6)

        warp = self._tps_warp(ordered_hull, resampled_target)
        vertices[front_idx, :2] = warp(front_xy)
        vertices[back_idx, :2] = warp(back_xy)
        lens_mesh.vertices = vertices
        return scale

    @staticmethod
    def _tps_warp(source_points: np.ndarray, target_points: np.ndarray):
        """Thin-plate-spline warp: exact at source_points -> target_points,
        smooth everywhere else. Falls back to a plain similarity transform
        (translate + uniform scale about the centroid) if there are too few
        distinct control points for a spline (degenerate hull)."""
        try:
            from scipy.interpolate import RBFInterpolator

            rbf = RBFInterpolator(source_points, target_points, kernel="thin_plate_spline")
            return lambda xy: rbf(xy)
        except Exception:
            src_center = source_points.mean(axis=0)
            dst_center = target_points.mean(axis=0)
            src_radius = float(np.linalg.norm(source_points - src_center, axis=1).mean())
            dst_radius = float(np.linalg.norm(target_points - dst_center, axis=1).mean())
            scale = dst_radius / max(src_radius, 1e-6)
            return lambda xy: dst_center + (xy - src_center) * scale

    @staticmethod
    def _hull_boundary(points_2d: np.ndarray) -> np.ndarray:
        """Outer 2D boundary of a vertex set. A thin boundary-loop lens's vertices
        ARE its hull (within float noise), so this is a no-op there; a solid/filled
        lens's hull is its true outer silhouette rather than every interior point."""
        if len(points_2d) < 3:
            return points_2d
        try:
            hull = ConvexHull(points_2d)
        except QhullError:
            # Degenerate (collinear / near-zero-area) point set -- fall back to
            # using every point rather than crashing.
            return points_2d
        return points_2d[hull.vertices]

    def _preserve_uvs(self, context: DeformationContext, selection: LensSelection) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        if hasattr(lens_mesh.visual, "uv") and lens_mesh.visual.uv is not None:
            lens_mesh.visual.uv = lens_mesh.visual.uv.copy()

    def _preserve_thickness(self, context: DeformationContext, selection: LensSelection, fit_scale: float) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)
        # Mean of each side separately, not an elementwise difference -- front_idx
        # and back_idx are not guaranteed equal length (true for a midpoint-based
        # split of an organic/asymmetric lens surface), so
        # lens_before[front_idx, 2] - lens_before[back_idx, 2] can raise a
        # broadcast ValueError (same bug class already fixed in RimDeformer's
        # equivalent _target_depth).
        #
        # Thickness is intentionally NOT rescaled by fit_scale: lens thickness is
        # a physical property of the lens blank, independent of how much its
        # footprint grows or shrinks to fit a different frame. fit_scale is
        # accepted here only so the signature matches _fit_lens_to_rim's return
        # value for callers that do need it; it is unused for thickness itself.
        del fit_scale
        original_thickness = float(np.mean(selection.lens_before[front_idx, 2]) - np.mean(selection.lens_before[back_idx, 2]))
        front_z = float(np.max(vertices[front_idx, 2]))
        vertices[front_idx, 2] = front_z
        vertices[back_idx, 2] = front_z - original_thickness
        lens_mesh.vertices = vertices

    def _preserve_curvature(self, context: DeformationContext, selection: LensSelection, fit_scale: float) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        original = selection.lens_before
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)
        # Not rescaled by fit_scale -- see _preserve_thickness: curvature, like
        # thickness, is a fixed physical property of the lens blank.
        del fit_scale
        original_front_profile = original[front_idx, 2] - np.mean(original[front_idx, 2])
        original_back_profile = original[back_idx, 2] - np.mean(original[back_idx, 2])
        vertices[front_idx, 2] = np.mean(vertices[front_idx, 2]) + original_front_profile
        vertices[back_idx, 2] = np.mean(vertices[back_idx, 2]) + original_back_profile
        lens_mesh.vertices = vertices

    def _validate(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> dict[str, Any]:
        lens_mesh = context.mesh(selection.lens_part)
        rim_mesh = context.mesh(selection.rim_part)
        lens_vertices = lens_mesh.vertices.copy()
        rim_vertices = rim_mesh.vertices.copy()
        front_idx = self._surface_indices(lens_vertices, front=True)
        back_idx = self._surface_indices(lens_vertices, front=False)
        # Use the outer hull, not every front vertex, for the same reason as
        # _fit_lens_to_rim: a solid/filled lens has interior vertices whose mean
        # distance-from-center is not the lens's actual radius, which previously
        # made "inside_rim" compare the wrong quantity against the rim boundary.
        front_hull = self._hull_boundary(lens_vertices[front_idx][:, :2])
        lens_boundary = front_hull[self._loop_order(front_hull)]
        fitted_target = self._resample_boundary(target_boundary, len(lens_boundary))
        fitted_target = self._align_boundary(lens_boundary, fitted_target)
        fit_error = float(np.mean(np.linalg.norm(lens_boundary - fitted_target, axis=1)))
        rim_front = rim_vertices[self._surface_indices(rim_vertices, front=True)][:, :2]
        rim_boundary = self._hull_boundary(rim_front)
        rim_center = rim_boundary.mean(axis=0)
        lens_center = lens_boundary.mean(axis=0)
        rim_radius = np.linalg.norm(rim_boundary - rim_center, axis=1).mean()
        lens_radius = np.linalg.norm(lens_boundary - lens_center, axis=1).mean()
        # Mean of each side separately -- see _preserve_thickness for why an
        # elementwise difference is unsafe here (front_idx/back_idx need not be
        # equal length).
        thickness = float(np.mean(lens_vertices[front_idx, 2]) - np.mean(lens_vertices[back_idx, 2]))
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
                "curvature_delta": round(float(np.std(lens_vertices[:, 2]) - np.std(selection.lens_before[:, 2])), 6),
                "inside_rim": bool(lens_radius < rim_radius),
                "valid": bool(fit_error < 2.5 and thickness > 0.0 and lens_radius < rim_radius),
            }
        )

    @staticmethod
    def _surface_indices(vertices: np.ndarray, front: bool) -> np.ndarray:
        z_values = vertices[:, 2]
        z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
        if np.isclose(z_min, z_max):
            # Degenerate: the whole lens sits at a single depth.
            idx = np.arange(len(z_values))
            return idx
        # A box-extruded lens (many vertices coplanar at the exact front/back faces)
        # and a curved/organic lens surface (only one or two vertices sit at the
        # exact float max/min Z, the rest vary continuously) both need a front/back
        # split. isclose-to-extreme only works for the former and silently starves
        # the split down to 1-2 vertices for the latter -- confirmed on
        # geometric_metal's lens (Cube.002_glass_mat_0): only 2 vertices matched the
        # exact Z extreme, so _preserve_thickness only ever repositioned those 2
        # vertices and left the other ~6900 at their tiny original z (~0.02mm),
        # which made the "inside_rim" validity check compare a 2-point lens
        # boundary against the real rim boundary and fail unpredictably depending
        # on which 2 points happened to win the float comparison. A midpoint
        # threshold (same fix already applied to RimDeformer's equivalent split for
        # the identical reason) always yields two full surfaces, regardless of
        # topology.
        midpoint = (z_min + z_max) / 2.0
        if front:
            return np.where(z_values >= midpoint)[0]
        return np.where(z_values < midpoint)[0]

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
