"""Rim-driven lens deformation that preserves topology, UVs, thickness, and curvature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate

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
            self._fit_lens_to_rim(context, selection, target_boundary)
            self._preserve_uvs(context, selection)
            self._preserve_thickness(context, selection)
            self._preserve_curvature(context, selection)
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
        boundary = rim_vertices[front_idx][:, :2]
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
    ) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)

        front_boundary = vertices[front_idx][:, :2]
        front_order = self._loop_order(front_boundary)
        back_boundary = vertices[back_idx][:, :2]
        back_order = self._loop_order(back_boundary)

        ordered_front_idx = front_idx[front_order]
        ordered_back_idx = back_idx[back_order]
        template_front = vertices[ordered_front_idx][:, :2]
        template_back = vertices[ordered_back_idx][:, :2]

        resampled_target = self._resample_boundary(target_boundary, len(ordered_front_idx))
        resampled_target = self._align_boundary(template_front, resampled_target)
        vertices[ordered_front_idx, :2] = resampled_target

        template_front_center = template_front.mean(axis=0)
        template_back_center = template_back.mean(axis=0)
        target_center = resampled_target.mean(axis=0)
        front_radius = np.linalg.norm(template_front - template_front_center, axis=1)
        target_radius = np.linalg.norm(resampled_target - target_center, axis=1)
        radial_scale = np.divide(target_radius, np.maximum(front_radius, 1e-6))
        back_relative = template_back - template_back_center
        vertices[ordered_back_idx, :2] = target_center + back_relative * radial_scale[:, None]
        lens_mesh.vertices = vertices

    def _preserve_uvs(self, context: DeformationContext, selection: LensSelection) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        if hasattr(lens_mesh.visual, "uv") and lens_mesh.visual.uv is not None:
            lens_mesh.visual.uv = lens_mesh.visual.uv.copy()

    def _preserve_thickness(self, context: DeformationContext, selection: LensSelection) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)
        original_thickness = float(np.mean(selection.lens_before[front_idx, 2] - selection.lens_before[back_idx, 2]))
        front_z = float(np.max(vertices[front_idx, 2]))
        vertices[front_idx, 2] = front_z
        vertices[back_idx, 2] = front_z - original_thickness
        lens_mesh.vertices = vertices

    def _preserve_curvature(self, context: DeformationContext, selection: LensSelection) -> None:
        lens_mesh = context.mesh(selection.lens_part)
        vertices = lens_mesh.vertices.copy()
        original = selection.lens_before
        front_idx = self._surface_indices(vertices, front=True)
        back_idx = self._surface_indices(vertices, front=False)
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
        lens_boundary = lens_vertices[front_idx][:, :2][self._loop_order(lens_vertices[front_idx][:, :2])]
        fitted_target = self._resample_boundary(target_boundary, len(lens_boundary))
        fitted_target = self._align_boundary(lens_boundary, fitted_target)
        fit_error = float(np.mean(np.linalg.norm(lens_boundary - fitted_target, axis=1)))
        rim_boundary = rim_vertices[self._surface_indices(rim_vertices, front=True)][:, :2]
        rim_center = rim_boundary.mean(axis=0)
        lens_center = lens_boundary.mean(axis=0)
        rim_radius = np.linalg.norm(rim_boundary - rim_center, axis=1).mean()
        lens_radius = np.linalg.norm(lens_boundary - lens_center, axis=1).mean()
        thickness = float(np.mean(lens_vertices[front_idx, 2] - lens_vertices[back_idx, 2]))
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
        z_target = float(np.max(z_values) if front else np.min(z_values))
        return np.where(np.isclose(z_values, z_target))[0]

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
