"""Quality checker for the final deformation engine output.

Scores the resulting mesh context across multiple axes:
* Geometry (bounding boxes, thickness)
* Constraints (failed/passed upstream constraint solver)
* Symmetry (left/right alignment)
* Fit (measurements matching target)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from backend.deformer.deformation_context import DeformationContext
from backend.geometry_units import m_to_mm


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

    def evaluate(self, context: DeformationContext) -> QualityReport:
        report = QualityReport()

        # Geometry score
        geom_score = self._score_geometry(context)
        report.breakdown["geometry"] = geom_score

        # Constraints score
        constraint_score = self._score_constraints(context)
        report.breakdown["constraints"] = constraint_score

        # Symmetry score
        sym_score = self._score_symmetry(context)
        report.breakdown["symmetry"] = sym_score

        # Physical dimension fit score
        fit_score, fit_warnings = self._score_fit(context)
        report.breakdown["fit"] = fit_score
        report.warnings.extend(fit_warnings)

        # Aggregate score
        weights = {"geometry": 0.2, "constraints": 0.25, "symmetry": 0.2, "fit": 0.35}
        total = sum(score * weights[k] for k, score in report.breakdown.items())
        report.score = total
        
        if total < 80.0:
            report.passed = False
            report.warnings.append(f"Overall quality score too low: {total:.1f}/100")
            
        if constraint_score < 70.0:
            report.passed = False
            report.warnings.append("Critical constraint failures detected.")

        topology_warnings = self._topology_warnings(context)
        if topology_warnings:
            report.passed = False
            report.warnings.extend(topology_warnings)

        return report

    @staticmethod
    def _topology_warnings(context: DeformationContext) -> list[str]:
        """Return critical geometry integrity problems after deformation."""
        warnings: list[str] = []
        required = (
            "Bridge",
            "LeftRim",
            "RightRim",
            "LeftLens",
            "RightLens",
            "LeftTemple",
            "RightTemple",
        )
        for name in required:
            try:
                mesh = context.mesh(name)
            except KeyError:
                warnings.append(f"{name} is missing from the final deformation context.")
                continue

            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if not np.isfinite(vertices).all():
                warnings.append(f"{name} contains non-finite vertex positions.")

            if len(faces):
                triangles = vertices[faces]
                double_area = np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                )
                degenerate_count = int(np.count_nonzero(double_area <= 1e-12))
                if degenerate_count:
                    warnings.append(
                        f"{name} contains {degenerate_count} degenerate triangles after deformation."
                    )

            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if len(normals):
                lengths = np.linalg.norm(normals, axis=1)
                if not np.isfinite(normals).all():
                    warnings.append(f"{name} contains non-finite vertex normals.")
                zero_count = int(np.count_nonzero(lengths <= 1e-12))
                if zero_count:
                    warnings.append(
                        f"{name} contains {zero_count} zero-length vertex normals."
                    )
        return warnings

    def _score_geometry(self, context: DeformationContext) -> float:
        score = 100.0
        try:
            frame = context.mesh("Frame")
            bounds = frame.bounds
            extents_mm = m_to_mm(bounds[1] - bounds[0])
            if extents_mm[0] > 200 or extents_mm[0] < 50:
                score -= 30.0
        except KeyError:
            score -= 50.0
        return max(0.0, score)

    def _score_constraints(self, context: DeformationContext) -> float:
        score = 100.0
        solver_meta = context.metadata.get("constraint_solver", {})
        failures = solver_meta.get("failures", [])
        corrections = solver_meta.get("corrections", [])
        
        score -= len(failures) * 20.0
        score -= len(corrections) * 5.0
        
        return max(0.0, score)

    def _score_fit(self, context: DeformationContext) -> tuple[float, list[str]]:
        m = context.measurements
        bridge = context.mesh("Bridge")
        left_rim = context.mesh("LeftRim")
        right_rim = context.mesh("RightRim")
        left_lens = context.mesh("LeftLens")
        right_lens = context.mesh("RightLens")

        measured = {
            "frame_width": m_to_mm(
                max(right_rim.bounds[1, 0], bridge.bounds[1, 0])
                - min(left_rim.bounds[0, 0], bridge.bounds[0, 0])
            ),
            "bridge_width": m_to_mm(bridge.extents[0]),
            "left_lens_width": m_to_mm(left_lens.extents[0]),
            "right_lens_width": m_to_mm(right_lens.extents[0]),
            "left_lens_height": m_to_mm(left_lens.extents[1]),
            "right_lens_height": m_to_mm(right_lens.extents[1]),
        }

        frame_vertices = np.vstack([
            left_rim.vertices,
            bridge.vertices,
            right_rim.vertices,
        ])

        def temple_length(name: str) -> float:
            temple = context.mesh(name)
            tree = cKDTree(frame_vertices)
            distances, _ = tree.query(temple.vertices, k=1)
            hinge = temple.vertices[int(np.argmin(distances))]
            return m_to_mm(float(np.max(np.linalg.norm(temple.vertices - hinge, axis=1))))

        measured["left_temple_length"] = temple_length("LeftTemple")
        measured["right_temple_length"] = temple_length("RightTemple")
        targets = {
            "frame_width": m.frame_width,
            "bridge_width": m.bridge_width,
            "left_lens_width": m.lens_width,
            "right_lens_width": m.lens_width,
            "left_lens_height": m.lens_height,
            "right_lens_height": m.lens_height,
            "left_temple_length": m.temple_length,
            "right_temple_length": m.temple_length,
        }
        tolerances = {
            "frame_width": 2.0,
            "bridge_width": 1.5,
            "left_lens_width": 1.5,
            "right_lens_width": 1.5,
            "left_lens_height": 1.5,
            "right_lens_height": 1.5,
            "left_temple_length": 2.0,
            "right_temple_length": 2.0,
        }

        penalties = []
        warnings: list[str] = []
        for key, target in targets.items():
            error = abs(float(measured[key]) - float(target))
            tolerance = tolerances[key]
            penalties.append(min(error / tolerance, 2.0))
            if error > tolerance:
                warnings.append(
                    f"{key} physical fit error {error:.2f} mm exceeds {tolerance:.2f} mm tolerance."
                )

        average_penalty = float(np.mean(penalties)) if penalties else 0.0
        score = max(0.0, 100.0 - average_penalty * 50.0)
        return score, warnings

    def _score_symmetry(self, context: DeformationContext) -> float:
        score = 100.0
        sym_meta = context.metadata.get("symmetry_solver", {})
        max_error = sym_meta.get("max_error_mm", 0.0)
        
        if max_error > 2.0:
            score -= 40.0
        elif max_error > 0.5:
            score -= 15.0
            
        return max(0.0, score)
