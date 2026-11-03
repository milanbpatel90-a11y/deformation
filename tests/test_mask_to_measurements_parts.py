from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.measurement.mask_to_measurements import MaskToMeasurements


def _blank_masks(height=80, width=120):
    return {class_id: np.zeros((height, width), dtype=np.uint8) for class_id in range(6)}


def test_frame_width_uses_both_disconnected_rim_components():
    masks = _blank_masks()
    masks[MaskToMeasurements.CLASS_RIM][10:20, 10:30] = 255
    masks[MaskToMeasurements.CLASS_RIM][10:20, 70:90] = 255

    converter = MaskToMeasurements(reference_width_mm=140.0)
    assert converter._measure_frame_width(masks, mm_per_px=1.0) == 80.0


def test_lens_dimensions_average_left_and_right_masks():
    masks = _blank_masks()
    masks[MaskToMeasurements.CLASS_LEFT_LENS][10:40, 10:30] = 255  # 20 x 30
    masks[MaskToMeasurements.CLASS_RIGHT_LENS][12:38, 70:94] = 255  # 24 x 26

    converter = MaskToMeasurements(reference_width_mm=140.0)
    width, height = converter._measure_lens_dimensions(masks, mm_per_px=1.0)
    assert width == 22.0
    assert height == 28.0


def test_class_mask_extraction_rejects_missing_left_or_right_lens():
    masks = _blank_masks()
    masks[MaskToMeasurements.CLASS_RIM][10:20, 10:90] = 255
    masks[MaskToMeasurements.CLASS_BRIDGE][10:16, 48:54] = 255
    masks[MaskToMeasurements.CLASS_RIGHT_LENS][20:50, 65:90] = 255

    converter = MaskToMeasurements(reference_width_mm=140.0)
    with pytest.raises(ValueError, match="missing non-empty left_lens"):
        converter.extract_from_class_masks(
            np.zeros((80, 120, 3), dtype=np.uint8), masks
        )


def test_class_mask_extraction_rejects_wrong_mask_dimensions():
    masks = _blank_masks()
    masks[MaskToMeasurements.CLASS_LEFT_LENS] = np.ones((20, 20), dtype=np.uint8)
    converter = MaskToMeasurements(reference_width_mm=140.0)
    with pytest.raises(ValueError, match="must have shape"):
        converter.extract_from_class_masks(
            np.zeros((80, 120, 3), dtype=np.uint8), masks
        )


def test_class_mask_extraction_uses_full_frame_extent():
    masks = _blank_masks()
    masks[MaskToMeasurements.CLASS_RIM][10:18, 10:35] = 255
    masks[MaskToMeasurements.CLASS_RIM][10:18, 75:100] = 255
    masks[MaskToMeasurements.CLASS_BRIDGE][12:18, 45:65] = 255
    masks[MaskToMeasurements.CLASS_LEFT_LENS][20:50, 12:40] = 255
    masks[MaskToMeasurements.CLASS_RIGHT_LENS][20:50, 70:98] = 255
    # A visible temple mask is available; nose pads are legitimately absent.
    masks[MaskToMeasurements.CLASS_TEMPLE][55:70, 10:14] = 255

    converter = MaskToMeasurements(reference_width_mm=140.0)
    measurements, contours = converter.extract_from_class_masks(
        np.zeros((80, 120, 3), dtype=np.uint8), masks
    )
    assert measurements.frame_width == 140.0
    assert len(contours.left) >= 4
    assert len(contours.right) >= 4
