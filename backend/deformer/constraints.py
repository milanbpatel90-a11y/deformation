"""Constraint solver for post-deformation correction and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


@dataclass(frozen=True)
class ConstraintIssue:
    """One warning, failure, or correction emitted by the solver."""

    stage: str
    code: str
    message: str
    value: float | bool | None = None
    target: float | bool | None = None


@dataclass
class ConstraintReport:
    """Summary emitted by the constraint solver."""

    passed: bool = True
    warnings: list[ConstraintIssue] = field(default_factory=list)
    corrections: list[ConstraintIssue] = field(default_factory=list)
    failures: list[ConstraintIssue] = field(default_factory=list)
    stages: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "warnings": [asdict(item) for item in self.warnings],
            "corrections": [asdict(item) for item in self.corrections],
            "failures": [asdict(item) for item in self.failures],
            "stages": self.stages,
        }


class ConstraintSolver(BaseDeformer):
    """Consume deformer stage metadata and apply small, safe corrections."""

    stage_name = "constraint_solver"

    def apply(self, context: DeformationContext) -> DeformationContext:
        report = ConstraintReport()
        report.stages["rim"] = self._solve_rim_constraints(context, report)
        report.stages["bridge"] = self._solve_bridge_constraints(context, report)
        report.stages["temple"] = self._solve_temple_constraints(context, report)
        report.stages["lens"] = self._solve_lens_constraints(context, report)
        report.stages["global"] = self._solve_global_constraints(context, report)
        report.passed = len(report.failures) == 0
        return self.update_context(context, **report.to_dict())

    def _solve_rim_constraints(self, context: DeformationContext, report: ConstraintReport) -> dict[str, Any]:
        rim = dict(context.metadata.get("rim_deformation", {}))
        min_thickness = float(context.descriptor.constraints.get("rim_thickness", {}).get("min", 0.8))
        max_thickness = float(context.descriptor.constraints.get("rim_thickness", {}).get("max", 6.0))
        current = float(context.measurements.rim_thickness)
        corrected = float(np.clip(current, min_thickness, max_thickness))
        if not np.isclose(current, corrected):
            context.measurements.rim_thickness = corrected
            report.corrections.append(
                ConstraintIssue(
                    stage="rim",
                    code="rim_thickness_clamped",
                    message="Rim thickness clamped to safe range.",
                    value=current,
                    target=corrected,
                )
            )
        rim["thickness_mm"] = round(corrected, 6)
        rim["valid"] = bool(rim.get("applied", False))
        return rim

    def _solve_bridge_constraints(self, context: DeformationContext, report: ConstraintReport) -> dict[str, Any]:
        bridge = dict(context.metadata.get("bridge_deformation", {}))
        if not bridge:
            report.failures.append(
                ConstraintIssue(stage="bridge", code="bridge_metadata_missing", message="Bridge metadata missing.")
            )
            return {"valid": False}

        width_limits = context.descriptor.constraints.get("bridge_width", {})
        minimum = float(width_limits.get("min", bridge.get("final_width", 0.0)))
        maximum = float(width_limits.get("max", bridge.get("final_width", 0.0)))
        final_width = float(bridge.get("final_width", 0.0))
        target_width = float(np.clip(final_width, minimum, maximum))

        if not np.isclose(final_width, target_width):
            self._scale_bridge_width(context, target_width)
            bridge["final_width"] = round(target_width, 6)
            report.corrections.append(
                ConstraintIssue(
                    stage="bridge",
                    code="bridge_width_clamped",
                    message="Bridge width clamped to constraint range.",
                    value=final_width,
                    target=target_width,
                )
            )

        lens_clearance = float(bridge.get("lens_clearance", 0.0))
        if lens_clearance < 0.0:
            corrected_width = max(minimum, target_width + lens_clearance - 0.5)
            self._scale_bridge_width(context, corrected_width)
            bridge["final_width"] = round(corrected_width, 6)
            bridge["lens_clearance"] = 0.5
            report.corrections.append(
                ConstraintIssue(
                    stage="bridge",
                    code="bridge_clearance_corrected",
                    message="Bridge width reduced to restore lens clearance.",
                    value=lens_clearance,
                    target=0.5,
                )
            )

        bridge["valid"] = True
        return bridge

    def _solve_temple_constraints(self, context: DeformationContext, report: ConstraintReport) -> dict[str, Any]:
        temple = dict(context.metadata.get("temple_deformation", {}))
        max_wrap = 35.0
        sides = []
        for side in temple.get("sides", []):
            current = dict(side)
            wrap_angle = float(current.get("wrap_angle", 0.0))
            hinge_error = float(current.get("hinge_error", 0.0))
            if abs(wrap_angle) > max_wrap:
                report.warnings.append(
                    ConstraintIssue(
                        stage="temple",
                        code="wrap_limit_exceeded",
                        message="Temple wrap exceeds preferred limit.",
                        value=wrap_angle,
                        target=max_wrap,
                    )
                )
                current["wrap_angle"] = float(np.clip(wrap_angle, -max_wrap, max_wrap))
            if hinge_error > 2.0:
                report.warnings.append(
                    ConstraintIssue(
                        stage="temple",
                        code="hinge_drift",
                        message="Temple hinge anchor drift is above preferred tolerance.",
                        value=hinge_error,
                        target=2.0,
                    )
                )
            current["valid"] = bool(current.get("valid", True))
            sides.append(current)
        temple["sides"] = sides
        temple["valid"] = all(side.get("valid", False) for side in sides) if sides else False
        return temple

    def _solve_lens_constraints(self, context: DeformationContext, report: ConstraintReport) -> dict[str, Any]:
        lens = dict(context.metadata.get("lens_deformation", {}))
        sides = []
        min_thickness = 0.4
        for side in lens.get("sides", []):
            current = dict(side)
            thickness = float(current.get("thickness_mm", 0.0))
            if thickness < min_thickness:
                self._restore_lens_thickness(context, current["side"], min_thickness)
                current["thickness_mm"] = min_thickness
                report.corrections.append(
                    ConstraintIssue(
                        stage="lens",
                        code="lens_thickness_restored",
                        message="Lens thickness restored to minimum safe thickness.",
                        value=thickness,
                        target=min_thickness,
                    )
                )
            if not bool(current.get("inside_rim", True)):
                self._shrink_lens_inside_rim(context, current["side"], factor=0.97)
                current["inside_rim"] = True
                report.corrections.append(
                    ConstraintIssue(
                        stage="lens",
                        code="lens_shrunk_inside_rim",
                        message="Lens slightly shrunk to fit inside rim boundary.",
                        value=False,
                        target=True,
                    )
                )
            current["valid"] = bool(current.get("valid", True))
            sides.append(current)
        lens["sides"] = sides
        lens["valid"] = all(side.get("valid", False) for side in sides) if sides else False
        return lens

    def _solve_global_constraints(self, context: DeformationContext, report: ConstraintReport) -> dict[str, Any]:
        frame = context.mesh("Frame")
        bounds = frame.bounds.astype(float)
        width = float(bounds[1, 0] - bounds[0, 0])
        height = float(bounds[1, 1] - bounds[0, 1])
        target_width = float(context.measurements.frame_width)
        width_error = abs(width - target_width)
        symmetry_error = abs(bounds[0, 0] + bounds[1, 0])

        if symmetry_error > 1.0:
            report.warnings.append(
                ConstraintIssue(
                    stage="global",
                    code="frame_off_center",
                    message="Frame centerline drift exceeds preferred tolerance.",
                    value=float(symmetry_error),
                    target=1.0,
                )
            )

        return {
            "frame_width_mm": round(width, 6),
            "frame_height_mm": round(height, 6),
            "frame_width_error_mm": round(width_error, 6),
            "symmetry_error_mm": round(float(symmetry_error), 6),
            "valid": True,
        }

    def _scale_bridge_width(self, context: DeformationContext, target_width: float) -> None:
        bridge_mesh = context.mesh("Bridge")
        vertices = bridge_mesh.vertices.copy()
        current_width = float(vertices[:, 0].max() - vertices[:, 0].min())
        if current_width <= 1e-6:
            return
        center_x = float(vertices[:, 0].mean())
        scale = target_width / current_width
        vertices[:, 0] = center_x + (vertices[:, 0] - center_x) * scale
        bridge_mesh.vertices = vertices

    def _restore_lens_thickness(self, context: DeformationContext, side: str, target_thickness: float) -> None:
        part = context.descriptor.vertex_groups[f"{side}_lens"][0]
        mesh = context.mesh(part)
        vertices = mesh.vertices.copy()
        front_z = float(np.max(vertices[:, 2]))
        back_z = float(np.min(vertices[:, 2]))
        current_thickness = front_z - back_z
        if current_thickness <= 1e-6:
            vertices[:, 2] = np.where(vertices[:, 2] >= front_z, front_z, front_z - target_thickness)
        else:
            scale = target_thickness / current_thickness
            center_z = (front_z + back_z) * 0.5
            vertices[:, 2] = center_z + (vertices[:, 2] - center_z) * scale
        mesh.vertices = vertices

    def _shrink_lens_inside_rim(self, context: DeformationContext, side: str, factor: float) -> None:
        part = context.descriptor.vertex_groups[f"{side}_lens"][0]
        mesh = context.mesh(part)
        vertices = mesh.vertices.copy()
        center = vertices[:, :2].mean(axis=0)
        vertices[:, :2] = center + (vertices[:, :2] - center) * factor
        mesh.vertices = vertices
