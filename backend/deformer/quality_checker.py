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
    critical: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "passed": self.passed,
            "warnings": self.warnings,
            "critical": self.critical,
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
        }


class QualityChecker:
    """Evaluate context and produce a final quality score."""

    def evaluate(self, context: DeformationContext) -> QualityReport:
        report = QualityReport()

        # Geometry score (also collects critical, score-independent failures)
        geom_score = self._score_geometry(context, report)
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

        # Critical geometry failures can never be averaged away by good symmetry /
        # constraint scores: a mangled mesh must not report success.
        if report.critical:
            report.passed = False
            report.warnings.extend(report.critical)

        return report

    # Plausibility limits for a real pair of glasses, in mm.
    BRIDGE_WIDTH_RANGE = (8.0, 40.0)
    FRAME_WIDTH_TOLERANCE = 0.15       # achieved vs requested front width
    MAX_RIM_TO_FRAME_RATIO = 0.62      # one rim can never be most of the frame
    LENS_OVERSHOOT_TOLERANCE_MM = 0.5  # lens may not exceed its rim's width by more than this

    def _score_geometry(self, context: DeformationContext, report: QualityReport | None = None) -> float:
        report = report if report is not None else QualityReport()
        score = 100.0

        # 1. Aliased parts: several logical parts are the same mesh object.
        for problem in context.distinct_part_problems():
            report.critical.append(f"Template not split into separate parts ({problem}).")
            score = 0.0

        # 2. Non-finite vertices anywhere.
        for name, mesh in context.meshes.items():
            if not np.isfinite(mesh.vertices).all():
                report.critical.append(f"Mesh '{name}' contains NaN/inf vertices.")
                score = 0.0

        try:
            frame = context.mesh("Frame")
            extents = frame.bounds[1] - frame.bounds[0]
            if extents[0] > 200 or extents[0] < 50:
                score -= 30.0
        except KeyError:
            return max(0.0, score - 50.0)

        # 3. Bridge must be bridge-sized, not frame-sized and not collapsed.
        bridge = context.meshes.get("Bridge")
        if bridge is not None:
            bridge_w = float(bridge.bounds[1][0] - bridge.bounds[0][0])
            lo, hi = self.BRIDGE_WIDTH_RANGE
            if not lo <= bridge_w <= hi:
                report.critical.append(
                    f"Bridge width {bridge_w:.1f} mm is outside the plausible {lo:.0f}-{hi:.0f} mm range."
                )
                score = min(score, 40.0)

        # 4. Front width (frame + rims + bridge, temples excluded) vs the request.
        front = [context.meshes[n] for n in ("Frame", "LeftRim", "RightRim", "Bridge") if n in context.meshes]
        front_w = float(
            np.max([m.bounds[1][0] for m in front]) - np.min([m.bounds[0][0] for m in front])
        )
        requested = float(getattr(context.measurements, "frame_width", 0.0) or 0.0)
        if requested > 0 and abs(front_w - requested) / requested > self.FRAME_WIDTH_TOLERANCE:
            report.warnings.append(
                f"Front width {front_w:.1f} mm differs from requested {requested:.1f} mm by more than "
                f"{self.FRAME_WIDTH_TOLERANCE:.0%}."
            )
            score -= 25.0

        # 5. A single rim spanning most of the frame means parts were not separated.
        for name in ("LeftRim", "RightRim"):
            rim = context.meshes.get(name)
            if rim is not None and front_w > 0:
                ratio = float(rim.bounds[1][0] - rim.bounds[0][0]) / front_w
                if ratio > self.MAX_RIM_TO_FRAME_RATIO:
                    report.critical.append(f"{name} spans {ratio:.0%} of the frame width.")
                    score = min(score, 40.0)

        # 6. A lens must sit inside its own rim. Warning only (score unchanged): the lens
        #    deformer is known to overshoot on curved lens meshes, and that must be visible
        #    in the report rather than silent.
        for side in ("Left", "Right"):
            lens, rim = context.meshes.get(f"{side}Lens"), context.meshes.get(f"{side}Rim")
            if lens is None or rim is None:
                continue
            lens_w = float(lens.bounds[1][0] - lens.bounds[0][0])
            rim_w = float(rim.bounds[1][0] - rim.bounds[0][0])
            if lens_w > rim_w + self.LENS_OVERSHOOT_TOLERANCE_MM:
                report.warnings.append(
                    f"{side}Lens ({lens_w:.1f} mm) is wider than its rim ({rim_w:.1f} mm): "
                    "the lens pokes outside the frame."
                )

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
