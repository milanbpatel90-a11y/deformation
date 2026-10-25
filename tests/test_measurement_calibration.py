import cv2
import numpy as np
import pytest

from backend.measurement.extractor import MeasurementExtractor
from backend.models import FrameMaterial, FrameShape


def _sample_image_and_mask():
    image = np.full((160, 400, 3), 255, dtype=np.uint8)
    mask = np.zeros((160, 400), dtype=np.uint8)
    cv2.rectangle(mask, (100, 55), (299, 105), 255, thickness=-1)
    return image, mask


def test_known_frame_width_is_used_as_calibration_reference():
    image, mask = _sample_image_and_mask()
    measurements, _ = MeasurementExtractor().extract_from_images(
        image,
        mask=mask,
        shape=FrameShape.RECTANGLE,
        material=FrameMaterial.METAL,
        reference_width_mm=142.0,
    )
    assert measurements.frame_width == 142.0


def test_rejects_implausible_calibration_width():
    image, mask = _sample_image_and_mask()
    with pytest.raises(ValueError, match="between 100 and 180"):
        MeasurementExtractor().extract_from_images(image, mask=mask, reference_width_mm=220.0)


def test_default_width_remains_compatible():
    image, mask = _sample_image_and_mask()
    measurements, _ = MeasurementExtractor().extract_from_images(image, mask=mask)
    assert measurements.frame_width == 135.0
