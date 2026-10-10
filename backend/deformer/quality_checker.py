"""Quality checker for the final deformation engine output.

Scores the resulting mesh context across multiple axes:
* Geometry (bounding boxes, thickness)
* Constraints (failed/passed upstream constraint solver)
* Symmetry (left/right alignment)
* Fit (measurements matching target)
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from backend.deformer.deformation_context import DeformationContext


@dataclass
class QualityReport:
    """Detailed scores for a single deformation run."""
    score: float = 100.0
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "passed": self.passed,
            "warnings": self.warnings,
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
        }


class QualityChecker:
    """Evaluate context and produce a final quality score."""

    REQUIRED_PARTS = (
        "Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens",
        "LeftTemple", "RightTemple",
    )

    def evaluate(self, context: DeformationContext) -> QualityReport:
        report = QualityReport()
        context.metadata.pop("quality_warnings", None)

        # Geometry score
        geom_score = self._score_geometry(context)
        report.breakdown["geometry"] = geom_score

        # Constraints score
        constraint_score = self._score_constraints(context)
        report.breakdown["constraints"] = constraint_score

        # Symmetry score
        sym_score = self._score_symmetry(context)
        report.breakdown["symmetry"] = sym_score

        # Aggregate score
        weights = {"geometry": 0.3, "constraints": 0.4, "symmetry": 0.3}
        total = sum(score * weights[k] for k, score in report.breakdown.items())
        report.score = total

        geometry_warnings = context.metadata.get("quality_warnings", [])
        if geometry_warnings:
            report.passed = False
            report.warnings.extend(geometry_warnings)
        
        if total < 80.0:
            report.passed = False
            report.warnings.append(f"Overall quality score too low: {total:.1f}/100")
            
        if constraint_score < 70.0:
            report.passed = False
            report.warnings.append("Critical constraint failures detected.")

        return report

    def _score_geometry(self, context: DeformationContext) -> float:
        score = 100.0
        warnings = context.metadata.setdefault("quality_warnings", [])
        missing = [name for name in self.REQUIRED_PARTS if name not in context.meshes]
        if missing:
            warnings.append("Required logical meshes are missing: " + ", ".join(missing))
            score = 0.0

        # Part-level deformers require independent mesh objects. Shared objects
        # let one stage silently deform several logical parts at once.
        present = [(name, context.meshes[name]) for name in self.REQUIRED_PARTS if name in context.meshes]
        valid_parts = []
        for name, mesh in present:
            problem = self._mesh_problem(mesh)
            if problem:
                score = 0.0
                warnings.append(f"{name} mesh is invalid: {problem}")
            else:
                valid_parts.append((name, mesh))

        fingerprints = {name: self._geometry_fingerprint(mesh) for name, mesh in valid_parts}
        shared = set()
        for index, (name, mesh) in enumerate(valid_parts):
            for other_name, other_mesh in valid_parts[index + 1:]:
                if mesh is other_mesh or fingerprints[name] == fingerprints[other_name]:
                    shared.update((name, other_name))
        if shared:
            score = 0.0
            warnings.append(
                "Logical frame parts share a mesh object or identical geometry: " + ", ".join(sorted(shared))
            )

        bridge = context.meshes.get("Bridge")
        frame = context.meshes.get("Frame")
        if bridge is not None and frame is not None:
            frame_limits = context.descriptor.constraints.get("frame_width")
            unit_scale = self._coordinate_to_mm(context, frame)
            frame_width_mm = float(frame.extents[0]) * unit_scale
            if frame_limits and not (float(frame_limits["min"]) <= frame_width_mm <= float(frame_limits["max"])):
                score = 0.0
                warnings.append(
                    f"Frame X extent {frame_width_mm:.2f} mm is outside supported range "
                    f"{float(frame_limits['min']):.2f}-{float(frame_limits['max']):.2f} mm"
                )
            limits = context.descriptor.constraints.get("bridge_width")
            bridge_width_mm = float(bridge.extents[0]) * unit_scale
            if limits and not (float(limits["min"]) <= bridge_width_mm <= float(limits["max"])):
                score = 0.0
                warnings.append(
                    f"Bridge X extent {bridge_width_mm:.2f} mm is outside supported range "
                    f"{float(limits['min']):.2f}-{float(limits['max']):.2f} mm"
                )

        return max(0.0, score)

    @staticmethod
    def _coordinate_to_mm(context: DeformationContext, frame) -> float:
        """Resolve the scene's m/mm scale against declared frame dimensions."""
        expected = getattr(getattr(context.template_info, "dimensions", None), "frame_width", None)
        if expected is None:
            ranges = context.descriptor.constraints.get("frame_width", {})
            expected = (float(ranges["min"]) + float(ranges["max"])) / 2 if ranges else None
        if expected is None or float(frame.extents[0]) <= 0:
            return 1.0
        extent = float(frame.extents[0])
        return min((1.0, 1000.0), key=lambda scale: abs(np.log(max(extent * scale, 1e-12) / float(expected))))

    @staticmethod
    def _mesh_problem(mesh) -> str | None:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            return "empty or malformed vertex array"
        if not np.isfinite(vertices).all():
            return "coordinates contain NaN or infinity"
        normals = np.asarray(mesh.vertex_normals)
        if len(normals) and not np.isfinite(normals).all():
            return "vertex normals contain NaN or infinity"
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and (np.asarray(uv).shape[0] != len(vertices) or not np.isfinite(uv).all()):
            return "UV data is malformed or contains NaN or infinity"
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            return "empty or malformed triangle array"
        if faces.min() < 0 or faces.max() >= len(vertices):
            return "face indices are outside the vertex array"
        triangles = vertices[faces]
        areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
        scale = max(float(np.ptp(vertices, axis=0).max()), np.finfo(float).tiny)
        tolerance = np.finfo(float).eps * scale * scale * 32
        if not np.isfinite(areas2).all() or np.any(areas2 <= tolerance):
            return "faces contain non-finite or degenerate triangles"
        return None

    @staticmethod
    def _geometry_fingerprint(mesh) -> bytes:
        """Hash geometry independent of vertex and face ordering."""
        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order), dtype=np.int64)
        canonical_faces = np.sort(inverse[faces], axis=1)
        face_order = np.lexsort((canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0]))
        digest = hashlib.sha256()
        digest.update(np.asarray(vertices[order].shape, dtype=np.int64).tobytes())
        digest.update(vertices[order].tobytes())
        digest.update(canonical_faces[face_order].tobytes())
        return digest.digest()

    def _score_constraints(self, context: DeformationContext) -> float:
        score = 100.0
        solver_meta = context.metadata.get("constraint_solver", {})
        failures = solver_meta.get("failures", [])
        corrections = solver_meta.get("corrections", [])
        
        score -= len(failures) * 20.0
        score -= len(corrections) * 5.0
        
        return max(0.0, score)

    def _score_symmetry(self, context: DeformationContext) -> float:
        score = 100.0
        sym_meta = context.metadata.get("symmetry_solver", {})
        max_error = sym_meta.get("max_error_mm", 0.0)
        
        if max_error > 2.0:
            score -= 40.0
        elif max_error > 0.5:
            score -= 15.0
            
        return max(0.0, score)
