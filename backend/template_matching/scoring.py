"""Weighted similarity scoring for eyewear template matching."""

from __future__ import annotations

from dataclasses import dataclass

from backend.models import BridgeType, FrameFamily, FrameMaterial, RimType, TemplateInfo
from backend.template_matching.feature_extractor import EyewearFeatureSet


@dataclass(frozen=True)
class ScoredTemplate:
    """One scored template candidate with a normalized 0-100 score."""

    template: TemplateInfo
    score: float
    breakdown: dict[str, float]
    similarities: dict[str, float]
    reason: str


class WeightedTemplateScorer:
    """Compare extracted features against a template profile using weighted similarity."""

    WEIGHTS = {
        "frame_family": 0.30,
        "lens_aspect_ratio": 0.20,
        "rim_type": 0.15,
        "bridge_type": 0.10,
        "material": 0.10,
        "frame_width": 0.07,
        "frame_height": 0.04,
        "temple_length": 0.02,
        "wrap_angle": 0.02,
        "confidence": 0.05,
    }

    def score(self, features: EyewearFeatureSet, template: TemplateInfo) -> ScoredTemplate:
        template_height = template.dimensions.lens_height / 0.75
        template_wrap = None

        similarities = {
            "frame_family": self._family_similarity(features.frame_family, template.profile.frame_family),
            "lens_aspect_ratio": self._numeric_similarity(
                features.lens_aspect_ratio, template.profile.lens_aspect_ratio, tolerance=0.45
            ),
            "rim_type": self._rim_similarity(features.rim_type, template.profile.rim_type),
            "bridge_type": self._bridge_similarity(features.bridge_type, template.profile.bridge_type),
            "material": self._material_similarity(features.material, template.profile.material),
            "frame_width": self._numeric_similarity(features.frame_width, template.dimensions.frame_width, tolerance=28.0),
            "frame_height": self._numeric_similarity(features.frame_height, template_height, tolerance=18.0),
            "temple_length": self._numeric_similarity(
                features.temple_length, template.dimensions.temple_length, tolerance=22.0
            ),
            "wrap_angle": self._optional_numeric_similarity(features.wrap_angle, template_wrap, default=0.5, tolerance=20.0),
        }
        similarities["confidence"] = self._confidence_similarity(features, similarities)

        breakdown = {
            key: round(self.WEIGHTS[key] * similarities[key] * 100.0, 2)
            for key in self.WEIGHTS
        }
        total = round(sum(breakdown.values()), 2)
        reason = self._reason(similarities)

        return ScoredTemplate(
            template=template,
            score=total,
            breakdown=breakdown,
            similarities={key: round(value, 4) for key, value in similarities.items()},
            reason=reason,
        )

    @staticmethod
    def _numeric_similarity(actual: float, expected: float, tolerance: float) -> float:
        return max(0.0, 1.0 - (abs(actual - expected) / max(tolerance, 1e-6)))

    @staticmethod
    def _optional_numeric_similarity(
        actual: float | None,
        expected: float | None,
        *,
        default: float,
        tolerance: float,
    ) -> float:
        if actual is None or expected is None:
            return default
        return WeightedTemplateScorer._numeric_similarity(actual, expected, tolerance)

    @staticmethod
    def _family_similarity(actual: FrameFamily, expected: FrameFamily) -> float:
        if actual == expected:
            return 1.0
        rectangular = {FrameFamily.WAYFARER, FrameFamily.RECTANGLE, FrameFamily.SQUARE, FrameFamily.OVERSIZED}
        if actual in rectangular and expected in rectangular:
            return 0.45
        if FrameFamily.GEOMETRIC in {actual, expected}:
            return 0.25
        return 0.0

    @staticmethod
    def _rim_similarity(actual: RimType, expected: RimType) -> float:
        if actual == expected:
            return 1.0
        if {actual, expected} == {RimType.FULL_RIM, RimType.SEMI_RIMLESS}:
            return 0.35
        return 0.0

    @staticmethod
    def _bridge_similarity(actual: BridgeType, expected: BridgeType) -> float:
        if actual == expected:
            return 1.0
        related = {
            frozenset({BridgeType.KEYHOLE, BridgeType.SADDLE}): 0.4,
            frozenset({BridgeType.PAD, BridgeType.DOUBLE}): 0.4,
            frozenset({BridgeType.PAD, BridgeType.STRAIGHT}): 0.2,
        }
        return related.get(frozenset({actual, expected}), 0.0)

    @staticmethod
    def _material_similarity(actual: FrameMaterial, expected: FrameMaterial) -> float:
        if actual == expected:
            return 1.0
        if {actual, expected} <= {FrameMaterial.ACETATE, FrameMaterial.PLASTIC, FrameMaterial.MIXED}:
            return 0.6
        if {actual, expected} <= {FrameMaterial.METAL, FrameMaterial.TITANIUM, FrameMaterial.MIXED}:
            return 0.7
        return 0.1 if FrameMaterial.MIXED in {actual, expected} else 0.0

    @staticmethod
    def _confidence_similarity(features: EyewearFeatureSet, similarities: dict[str, float]) -> float:
        primary = (
            similarities["frame_family"]
            + similarities["rim_type"]
            + similarities["bridge_type"]
            + similarities["material"]
        ) / 4.0
        return min(1.0, max(0.0, features.confidence) * primary)

    @staticmethod
    def _reason(similarities: dict[str, float]) -> str:
        strongest = sorted(
            (
                ("frame_family", similarities["frame_family"]),
                ("lens_aspect_ratio", similarities["lens_aspect_ratio"]),
                ("rim_type", similarities["rim_type"]),
                ("bridge_type", similarities["bridge_type"]),
                ("material", similarities["material"]),
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        labels = "+".join(name for name, score in strongest if score >= 0.75)
        return labels or "closest_overall_match"
