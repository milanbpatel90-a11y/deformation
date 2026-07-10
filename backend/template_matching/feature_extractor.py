"""Convert segmented eyewear images plus measurements into matchable features."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from backend.models import BridgeType, FrameFamily, FrameMaterial, Measurements, RimType, StyleClassification


class EyewearFeatureSet(BaseModel):
    """Semantic and geometric features used for template selection."""

    frame_family: FrameFamily
    rim_type: RimType
    bridge_type: BridgeType
    material: FrameMaterial
    lens_aspect_ratio: float
    frame_width: float
    frame_height: float
    temple_length: float
    wrap_angle: float | None = None
    confidence: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)


class FeatureExtractor:
    """Build a stable feature set from segmentation output and measurements."""

    def extract(
        self,
        front_mask: np.ndarray | None,
        measurements: Measurements,
        style: StyleClassification,
        top: np.ndarray | None = None,
    ) -> EyewearFeatureSet:
        frame_height = self._estimate_frame_height(front_mask, measurements)
        wrap_angle = self._estimate_wrap_angle(top)
        metrics = {
            "source": "segmentation+measurements",
            "shape": style.shape.value,
            "nose_pads": style.nose_pads,
            "classifier_confidence": style.confidence,
        }
        metrics.update(style.metrics)
        if wrap_angle is not None:
            metrics["wrap_angle_estimated"] = True

        return EyewearFeatureSet(
            frame_family=style.frame_family,
            rim_type=style.rim_type,
            bridge_type=style.bridge_type,
            material=style.material,
            lens_aspect_ratio=round(measurements.lens_width / max(measurements.lens_height, 1e-6), 4),
            frame_width=round(measurements.frame_width, 2),
            frame_height=round(frame_height, 2),
            temple_length=round(measurements.temple_length, 2),
            wrap_angle=round(wrap_angle, 2) if wrap_angle is not None else None,
            confidence=float(style.confidence),
            metrics=metrics,
        )

    def from_measurements(
        self,
        measurements: Measurements,
        style: StyleClassification,
    ) -> EyewearFeatureSet:
        return EyewearFeatureSet(
            frame_family=style.frame_family,
            rim_type=style.rim_type,
            bridge_type=style.bridge_type,
            material=style.material,
            lens_aspect_ratio=round(measurements.lens_width / max(measurements.lens_height, 1e-6), 4),
            frame_width=round(measurements.frame_width, 2),
            frame_height=round(measurements.lens_height / 0.75, 2),
            temple_length=round(measurements.temple_length, 2),
            wrap_angle=None,
            confidence=float(style.confidence),
            metrics={"source": "measurements"},
        )

    @staticmethod
    def _estimate_frame_height(front_mask: np.ndarray | None, measurements: Measurements) -> float:
        if front_mask is None:
            return measurements.lens_height / 0.75

        contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return measurements.lens_height / 0.75

        contour = max(contours, key=cv2.contourArea)
        _x, _y, width_px, height_px = cv2.boundingRect(contour)
        mm_per_px = measurements.frame_width / max(width_px, 1)
        return max(measurements.lens_height, height_px * mm_per_px)

    @staticmethod
    def _estimate_wrap_angle(top: np.ndarray | None) -> float | None:
        if top is None:
            return None

        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 0:
            return None

        (_, _), (major_axis, minor_axis), _angle = cv2.minAreaRect(contour)
        if major_axis <= 0:
            return None

        curvature_ratio = 1.0 - min(minor_axis / major_axis, 1.0)
        return float(np.clip(curvature_ratio * 35.0, 0.0, 35.0))
