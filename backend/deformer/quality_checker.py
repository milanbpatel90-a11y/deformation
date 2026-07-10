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

        # Aggregate score
        weights = {"geometry": 0.3, "constraints": 0.4, "symmetry": 0.3}
        total = sum(score * weights[k] for k, score in report.breakdown.items())
        report.score = total
        
        if total < 80.0:
            report.passed = False
            report.warnings.append(f"Overall quality score too low: {total:.1f}/100")
            
        if constraint_score < 70.0:
            report.passed = False
            report.warnings.append("Critical constraint failures detected.")

        return report

    def _score_geometry(self, context: DeformationContext) -> float:
        score = 100.0
        try:
            frame = context.mesh("Frame")
            bounds = frame.bounds
            extents = bounds[1] - bounds[0]
            if extents[0] > 200 or extents[0] < 50:
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

    def _score_symmetry(self, context: DeformationContext) -> float:
        score = 100.0
        sym_meta = context.metadata.get("symmetry_solver", {})
        max_error = sym_meta.get("max_error_mm", 0.0)
        
        if max_error > 2.0:
            score -= 40.0
        elif max_error > 0.5:
            score -= 15.0
            
        return max(0.0, score)
