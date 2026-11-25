"""Turn approved/selected reference frames into fused, provenance-tagged
measurements, reusing the existing measurement/fusion code rather than
inventing a parallel estimator.

Calibration: a video alone has no absolute scale reference. This module
supports the manual-entry calibration path in full (a merchant-supplied
known total frame width in mm rescales every linear measurement). A
reference-object/marker detector (the brief's stretch goal) is NOT
implemented -- see the package README's "Known limitations" section and
the `CalibrationInput` docstring. Without manual calibration, measurements
fall back to the existing `MeasurementExtractor`/`MeasurementFuser`
heuristic estimate (same one `DeformationPipeline.run_from_multiple_images`
already relies on for ordinary photo uploads) and are tagged
`calibration_status=estimated`, never silently relabeled as exact mm.
"""

from __future__ import annotations

import numpy as np

from backend.classifier.shape_classifier import ShapeClassifier
from backend.fusion.measurement_fuser import MeasurementFuser
from backend.measurement.extractor import MeasurementExtractor
from backend.models import Measurements
from backend.video_extraction.schemas import (
    CalibrationInput,
    CalibrationStatus,
    MeasurementsBundle,
    MeasurementValue,
    SelectedFrame,
    ViewLabel,
)

_VIDEO_VIEW_TO_FUSION_VIEW = {
    ViewLabel.FRONT: "front",
    ViewLabel.FRONT_LEFT_45: "three_quarter",
    ViewLabel.FRONT_RIGHT_45: "three_quarter",
    ViewLabel.LEFT_SIDE: "side",
    ViewLabel.RIGHT_SIDE: "side",
    ViewLabel.TOP: "top",
    ViewLabel.BOTTOM: "top",
    ViewLabel.REAR_INSIDE: "side",
    ViewLabel.UNKNOWN: "unknown",
}

_LINEAR_FIELDS = ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length", "rim_thickness")


class MeasurementPreprocessor:
    def __init__(self):
        self.classifier = ShapeClassifier()
        self.measurer = MeasurementExtractor()
        self.fuser = MeasurementFuser()

    def build(
        self,
        frames_with_masks: list[tuple[SelectedFrame, np.ndarray, np.ndarray]],
        calibration: CalibrationInput | None = None,
    ) -> MeasurementsBundle:
        """``frames_with_masks``: (SelectedFrame record, decoded BGR image,
        combined binary mask) for every approved frame, in any order."""
        warnings: list[str] = []
        if not frames_with_masks:
            return MeasurementsBundle(
                fused=None,
                per_measurement=[],
                calibration_status=CalibrationStatus.ESTIMATED,
                warnings=["No approved frames were available to derive measurements from."],
                sufficient_for_deformation=False,
            )

        per_view_measurements: list[tuple[str, Measurements]] = []
        per_measurement: list[MeasurementValue] = []

        for selected, image, mask in frames_with_masks:
            style = self.classifier.classify_style(image, mask)
            measurements, _contour = self.measurer.extract_from_images(
                image,
                side=None,
                mask=mask,
                shape=style.shape,
                material=style.material,
                nose_pads=style.nose_pads,
                color=measurements_color_default(),
            )
            fusion_view = _VIDEO_VIEW_TO_FUSION_VIEW.get(selected.view, "unknown")
            per_view_measurements.append((fusion_view, measurements))

            confidence = round(min(1.0, selected.composite_score / 100.0) * (0.6 + 0.4 * selected.detection.confidence), 3)
            frame_measurement_names: list[str] = []
            for field in _LINEAR_FIELDS:
                per_measurement.append(
                    MeasurementValue(
                        name=field,
                        value=float(getattr(measurements, field)),
                        unit="mm",
                        confidence=confidence,
                        source_view=selected.view.value,
                        calibration_status=CalibrationStatus.ESTIMATED,
                        source_frame_index=selected.frame_index,
                    )
                )
                frame_measurement_names.append(field)
            # Record, on the frame itself, which measurements it fed -- this is
            # the "measurement references" the brief asks each selected image
            # to carry (section 7's per-image field list).
            selected.measurement_refs = frame_measurement_names

        try:
            fused = self.fuser.fuse(per_view_measurements)
        except ValueError as exc:
            return MeasurementsBundle(
                fused=None,
                per_measurement=per_measurement,
                calibration_status=CalibrationStatus.ESTIMATED,
                warnings=[f"Could not fuse measurements: {exc}"],
                sufficient_for_deformation=False,
            )

        calibration_status = CalibrationStatus.ESTIMATED
        calibration_reference_mm = None
        if calibration and calibration.known_frame_width_mm:
            measured_width = float(fused.frame_width) or 1.0
            scale = calibration.known_frame_width_mm / measured_width
            fused = fused.model_copy(
                update={field: round(getattr(fused, field) * scale, 2) for field in _LINEAR_FIELDS}
            )
            calibration_status = CalibrationStatus.CALIBRATED
            calibration_reference_mm = calibration.known_frame_width_mm
            for item in per_measurement:
                item.calibration_status = CalibrationStatus.CALIBRATED
        else:
            warnings.append(
                "No calibration reference supplied: measurements are heuristic estimates "
                "(typical-product-photo framing assumption), not metrology-grade millimetres. "
                "Provide `known_frame_width_mm` for a calibrated result."
            )

        views_present = {v for v, _ in per_view_measurements}
        if "front" not in views_present:
            warnings.append("No clean front view was selected; frame/lens/bridge width estimates are less reliable.")
        if "side" not in views_present:
            warnings.append("No side view was selected; temple length/curvature fell back to a frame-width heuristic.")
        if "top" not in views_present:
            warnings.append("No top view was selected; rim thickness fell back to a frame-width heuristic.")

        return MeasurementsBundle(
            fused=fused.model_dump(),
            per_measurement=per_measurement,
            calibration_status=calibration_status,
            calibration_reference_mm=calibration_reference_mm,
            warnings=warnings,
            sufficient_for_deformation=True,
        )


def measurements_color_default() -> str:
    return "#d9a7a2"
