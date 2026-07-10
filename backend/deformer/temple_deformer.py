"""Hinge-driven temple deformation for length, wrap, ear bend, and thickness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


@dataclass(frozen=True)
class TempleSelection:
    """Temple-local mesh selection for one side."""

    side: str
    temple_part: str
    tip_part: str | None
    hinge_pivot: np.ndarray
    axis: np.ndarray
    temple_before: np.ndarray
    tip_before: np.ndarray | None


class TempleDeformer(BaseDeformer):
    """Rotate and extend temples from hinge pivots without moving the frame."""

    stage_name = "temple_deformation"

    def __init__(self, max_wrap_angle: float = 35.0, max_ear_bend: float = 28.0):
        self.max_wrap_angle = float(max_wrap_angle)
        self.max_ear_bend = float(max_ear_bend)

    def apply(self, context: DeformationContext) -> DeformationContext:
        results: list[dict[str, Any]] = []
        frame_before = context.mesh("Frame").vertices.copy()

        for side in ("left", "right"):
            selection = self._collect_temples(context, side)
            target = self._compute_target(context, selection)
            self._rotate_about_hinge(context, selection, target)
            self._extend_length(context, selection, target)
            self._apply_wrap(context, selection, target)
            self._apply_ear_bend(context, selection, target)
            self._preserve_symmetry(context, selection)
            results.append(self._validate_constraints(context, selection, target))

        frame_after = context.mesh("Frame").vertices.copy()
        frame_max_displacement = float(np.linalg.norm(frame_after - frame_before, axis=1).max())
        return self.update_context(
            context,
            applied=True,
            frame_max_displacement=round(frame_max_displacement, 6),
            sides=results,
        )

    def _collect_temples(self, context: DeformationContext, side: str) -> TempleSelection:
        temple_part = context.descriptor.vertex_groups[f"{side}_temple"][0]
        hinge = context.descriptor.hinges[side]
        tip_part = self._tip_part_for_side(context, side)
        return TempleSelection(
            side=side,
            temple_part=temple_part,
            tip_part=tip_part,
            hinge_pivot=hinge.pivot.astype(np.float64),
            axis=hinge.axis.astype(np.float64),
            temple_before=context.mesh(temple_part).vertices.copy(),
            tip_before=context.mesh(tip_part).vertices.copy() if tip_part else None,
        )

    def _compute_target(self, context: DeformationContext, selection: TempleSelection) -> dict[str, Any]:
        constraints = context.descriptor.constraints
        current_length = self._temple_length(selection.temple_before, selection.hinge_pivot, selection.axis)
        requested_length = float(context.measurements.temple_length)
        length_limits = constraints.get("temple_length", {"min": requested_length, "max": requested_length})
        target_length = float(np.clip(requested_length, length_limits["min"], length_limits["max"]))

        requested_wrap = float(context.measurements.temple_curve_angle - context.template_info.dimensions.temple_curve_angle)
        target_wrap = float(np.clip(requested_wrap, -self.max_wrap_angle, self.max_wrap_angle))
        ear_bend = float(np.clip(8.0 + max(target_length - current_length, 0.0) * 0.12, 4.0, self.max_ear_bend))
        thickness_scale = float(
            np.clip(
                context.measurements.rim_thickness / max(context.template_info.dimensions.rim_thickness, 1e-6),
                0.8,
                1.6,
            )
        )
        return {
            "current_length": current_length,
            "target_length": target_length,
            "requested_length": requested_length,
            "wrap_angle": target_wrap,
            "ear_bend": ear_bend,
            "thickness_scale": thickness_scale,
            "length_clamped": not np.isclose(requested_length, target_length),
        }

    def _rotate_about_hinge(self, context: DeformationContext, selection: TempleSelection, target: dict[str, Any]) -> None:
        sign = -1.0 if selection.side == "left" else 1.0
        base_angle = sign * min(target["wrap_angle"] * 0.35, self.max_wrap_angle * 0.35)
        rotation = Rotation.from_euler("y", base_angle, degrees=True)

        for part in filter(None, [selection.temple_part, selection.tip_part]):
            mesh = context.mesh(part)
            vertices = mesh.vertices.copy()
            weights = self._progress_along_axis(vertices, selection.hinge_pivot, selection.axis)
            rotated = np.array([selection.hinge_pivot + rotation.apply(vertex - selection.hinge_pivot) for vertex in vertices])
            mesh.vertices = vertices * (1.0 - weights[:, None]) + rotated * weights[:, None]

    def _extend_length(self, context: DeformationContext, selection: TempleSelection, target: dict[str, Any]) -> None:
        length_scale = target["target_length"] / max(target["current_length"], 1e-6)
        for part in filter(None, [selection.temple_part, selection.tip_part]):
            mesh = context.mesh(part)
            vertices = mesh.vertices.copy()
            weights = self._progress_along_axis(vertices, selection.hinge_pivot, selection.axis)
            axial = np.dot(vertices - selection.hinge_pivot, selection.axis)
            positive = np.maximum(axial, 0.0)
            extension = positive * (length_scale - 1.0) * weights
            vertices += selection.axis * extension[:, None]
            mesh.vertices = vertices

    def _apply_wrap(self, context: DeformationContext, selection: TempleSelection, target: dict[str, Any]) -> None:
        sign = -1.0 if selection.side == "left" else 1.0
        rotation = Rotation.from_euler("z", sign * target["wrap_angle"], degrees=True)
        mesh = context.mesh(selection.temple_part)
        vertices = mesh.vertices.copy()
        weights = self._progress_along_axis(vertices, selection.hinge_pivot, selection.axis)
        rotated = np.array([selection.hinge_pivot + rotation.apply(vertex - selection.hinge_pivot) for vertex in vertices])
        mesh.vertices = vertices * (1.0 - weights[:, None]) + rotated * weights[:, None]

    def _apply_ear_bend(self, context: DeformationContext, selection: TempleSelection, target: dict[str, Any]) -> None:
        for part in filter(None, [selection.temple_part, selection.tip_part]):
            mesh = context.mesh(part)
            vertices = mesh.vertices.copy()
            progress = self._progress_along_axis(vertices, selection.hinge_pivot, selection.axis)
            bend_zone = self.apply_falloff(np.clip((progress - 0.6) / 0.4, 0.0, 1.0))
            downward = -bend_zone * np.deg2rad(target["ear_bend"]) * 6.0
            backward = -bend_zone * np.deg2rad(target["ear_bend"]) * 3.0
            vertices[:, 1] += downward
            vertices[:, 2] += backward
            center_z = float(vertices[:, 2].mean())
            vertices[:, 2] = center_z + (vertices[:, 2] - center_z) * target["thickness_scale"]
            mesh.vertices = vertices

    def _preserve_symmetry(self, context: DeformationContext, selection: TempleSelection) -> None:
        sign = -1.0 if selection.side == "left" else 1.0
        mesh = context.mesh(selection.temple_part)
        vertices = mesh.vertices.copy()
        axial = np.dot(vertices - selection.hinge_pivot, selection.axis)
        positive = np.maximum(axial, 0.0)
        vertices[:, 0] = selection.hinge_pivot[0] + sign * np.abs(vertices[:, 0] - selection.hinge_pivot[0])
        vertices[:, 0] = np.where(positive <= 1e-6, selection.hinge_pivot[0], vertices[:, 0])
        mesh.vertices = vertices
        if selection.tip_part:
            tip_mesh = context.mesh(selection.tip_part)
            tip_vertices = tip_mesh.vertices.copy()
            tip_vertices[:, 0] = selection.hinge_pivot[0] + sign * np.abs(tip_vertices[:, 0] - selection.hinge_pivot[0])
            tip_mesh.vertices = tip_vertices

    def _validate_constraints(self, context: DeformationContext, selection: TempleSelection, target: dict[str, Any]) -> dict[str, Any]:
        temple_after = context.mesh(selection.temple_part).vertices.copy()
        final_length = self._temple_length(temple_after, selection.hinge_pivot, selection.axis)
        hinge_error = self._hinge_anchor_error(selection.temple_before, temple_after, selection.hinge_pivot)
        frame_bounds = context.mesh("Frame").bounds
        temple_bounds = context.mesh(selection.temple_part).bounds
        frame_clearance = float(np.min(np.abs(temple_bounds[:, 0][:, None] - frame_bounds[:, 0][None, :])))
        return self.validate_constraints(
            {
                "side": selection.side,
                "requested_length": round(target["requested_length"], 4),
                "target_length": round(target["target_length"], 4),
                "final_length": round(final_length, 4),
                "length_clamped": target["length_clamped"],
                "wrap_angle": round(target["wrap_angle"], 4),
                "ear_bend": round(target["ear_bend"], 4),
                "hinge_error": round(hinge_error, 6),
                "frame_clearance": round(frame_clearance, 6),
                "valid": hinge_error < 2.0 and frame_clearance >= 0.0,
            }
        )

    def _tip_part_for_side(self, context: DeformationContext, side: str) -> str | None:
        tips = context.descriptor.vertex_groups.get("temple_tips", [])
        if not tips:
            return None
        tip_mesh = context.mesh(tips[0])
        sign = -1.0 if side == "left" else 1.0
        if np.any(sign * tip_mesh.vertices[:, 0] > 0):
            return tips[0]
        return None

    @staticmethod
    def _progress_along_axis(vertices: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> np.ndarray:
        axial = np.dot(vertices - pivot, axis)
        positive = np.maximum(axial, 0.0)
        max_positive = max(float(positive.max()), 1e-6)
        return BaseDeformer.apply_falloff(np.clip(positive / max_positive, 0.0, 1.0))

    @staticmethod
    def _temple_length(vertices: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> float:
        axial = np.dot(vertices - pivot, axis)
        return float(np.max(axial) - np.min(axial))

    @staticmethod
    def _hinge_anchor_error(before: np.ndarray, after: np.ndarray, hinge_pivot: np.ndarray) -> float:
        distances = np.linalg.norm(before - hinge_pivot, axis=1)
        anchor_idx = np.argsort(distances)[:8]
        return float(np.linalg.norm(after[anchor_idx] - before[anchor_idx], axis=1).max())
