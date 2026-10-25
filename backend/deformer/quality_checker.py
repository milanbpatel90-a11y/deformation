"""Post-deformation quality checks for eyewear meshes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.deformer.deformation_context import DeformationContext


@dataclass
class QualityReport:
    score: float = 100.0
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "passed": self.passed,
            "warnings": self.warnings,
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
        }


class QualityChecker:
    """Evaluate deformed context and produce a quality score (0–100)."""

    FRAME_WIDTH_RANGE = (95.0, 185.0)
    BRIDGE_RANGE = (8.0, 32.0)

    def evaluate(self, context: DeformationContext) -> QualityReport:
        report = QualityReport()
        report.breakdown["geometry"] = self._score_geometry(context, report)
        report.breakdown["constraints"] = self._score_constraints(context, report)
        report.breakdown["symmetry"] = self._score_symmetry(context, report)
        report.breakdown["proportions"] = self._score_proportions(context, report)
        report.breakdown["mesh_integrity"] = self._score_mesh_integrity(context, report)

        weights = {
            "geometry": 0.2,
            "constraints": 0.25,
            "symmetry": 0.2,
            "proportions": 0.2,
            "mesh_integrity": 0.15,
        }
        total = sum(report.breakdown.get(k, 0.0) * w for k, w in weights.items())
        report.score = float(np.clip(total, 0.0, 100.0))

        if report.score < 70.0:
            report.passed = False
            report.warnings.append(f"Overall quality score too low: {report.score:.1f}/100")
        if report.breakdown.get("constraints", 100) < 60.0:
            report.passed = False
            report.warnings.append("Critical constraint failures detected.")
        # Missing required part meshes are a hard failure even if the weighted
        # average happens to clear the general score threshold.
        if report.breakdown.get("mesh_integrity", 100.0) < 100.0:
            report.passed = False
            report.warnings.append("Critical mesh integrity gate failed.")
        return report

    def _score_geometry(self, context: DeformationContext, report: QualityReport) -> float:
        score = 100.0
        try:
            frame = context.mesh("Frame")
            extents = frame.bounds[1] - frame.bounds[0]
            fw = float(extents[0])
            lo, hi = self.FRAME_WIDTH_RANGE
            if fw < lo or fw > hi:
                score -= 35.0
                report.warnings.append(f"Frame width {fw:.1f}mm outside [{lo}, {hi}]")
            elif fw < lo + 5 or fw > hi - 5:
                score -= 10.0
        except KeyError:
            score -= 50.0
            report.warnings.append("Missing Frame geometry")
        return max(0.0, score)

    def _score_constraints(self, context: DeformationContext, report: QualityReport) -> float:
        score = 100.0
        solver_meta = context.metadata.get("constraint_solver", {})
        failures = solver_meta.get("failures", [])
        corrections = solver_meta.get("corrections", [])
        score -= len(failures) * 18.0
        score -= len(corrections) * 4.0
        if failures:
            report.warnings.append(f"Constraint failures: {len(failures)}")
        return max(0.0, score)

    def _score_symmetry(self, context: DeformationContext, report: QualityReport) -> float:
        score = 100.0
        sym_meta = context.metadata.get("symmetry_solver", {})
        max_error = float(sym_meta.get("max_error_mm", 0.0))
        if max_error > 3.0:
            score -= 45.0
            report.warnings.append(f"Symmetry error {max_error:.2f}mm")
        elif max_error > 1.0:
            score -= 20.0
        elif max_error > 0.5:
            score -= 8.0
        return max(0.0, score)

    def _score_proportions(self, context: DeformationContext, report: QualityReport) -> float:
        score = 100.0
        try:
            left = context.mesh("LeftLens")
            right = context.mesh("RightLens")
            le = left.bounds[1] - left.bounds[0]
            re = right.bounds[1] - right.bounds[0]
            if abs(le[0] - re[0]) > 3.0:
                score -= 20.0
                report.warnings.append("Left/right lens width mismatch")
            for side, e in (("L", le), ("R", re)):
                ar = float(e[0]) / max(float(e[1]), 1e-6)
                if ar < 0.7 or ar > 3.5:
                    score -= 15.0
                    report.warnings.append(f"{side} lens aspect {ar:.2f} unusual")
        except KeyError:
            score -= 25.0
        try:
            bridge = context.mesh("Bridge")
            be = bridge.bounds[1] - bridge.bounds[0]
            bw = float(be[0])
            lo, hi = self.BRIDGE_RANGE
            if bw < lo or bw > hi:
                score -= 15.0
                report.warnings.append(f"Bridge width {bw:.1f}mm outside range")
        except KeyError:
            score -= 10.0
        return max(0.0, score)

    def _score_mesh_integrity(self, context: DeformationContext, report: QualityReport) -> float:
        score = 100.0
        # A production eyewear frame must retain the bridge, both rims, both
        # lenses and both temples. Missing one of these parts is not cosmetic.
        critical = [
            "Frame", "Bridge", "LeftRim", "RightRim",
            "LeftLens", "RightLens", "LeftTemple", "RightTemple",
        ]
        missing = 0
        empty = 0
        for name in critical:
            try:
                m = context.mesh(name)
                if len(getattr(m, "faces", [])) < 1 or len(getattr(m, "vertices", [])) < 3:
                    empty += 1
            except KeyError:
                missing += 1
        score -= missing * 20.0
        score -= empty * 15.0
        if missing:
            report.warnings.append(f"Missing critical parts: {missing}")
        if empty:
            report.warnings.append(f"Empty critical parts: {empty}")
        return max(0.0, score)
