"""Compatibility adapter for the repository's one-class eyewear segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from backend.measurement.extractor import MeasurementExtractor
from backend.models import FrameMaterial, FrameShape, LensContour, Measurements


class MaskToMeasurements:
    """Convert a one-class binary eyewear mask into the real measurement model.

    This module is retained for compatibility with older callers. It no longer
    assumes six semantic YOLO part classes; the production model is binary
    eyewear-vs-background and measurement extraction is delegated to the same
    MeasurementExtractor used by DeformationPipeline.
    """

    DEFAULT_FRAME_WIDTH_MM = 140.0

    def __init__(self, reference_width_mm: float = DEFAULT_FRAME_WIDTH_MM):
        self.reference_width_mm = reference_width_mm
        self._extractor = MeasurementExtractor()
        self._extractor.DEFAULT_FRAME_WIDTH_MM = reference_width_mm

    def extract_from_yolo_results(
        self,
        image: np.ndarray,
        results,
        shape: FrameShape = FrameShape.GEOMETRIC,
        material: FrameMaterial = FrameMaterial.METAL,
        color: str = "#000000",
    ) -> tuple[Measurements, LensContour]:
        mask = self._union_binary_masks(results, image.shape[:2])
        if mask is None:
            raise ValueError("No binary eyewear mask was returned by the segmentation model")
        return self._extractor.extract_from_images(
            image,
            mask=mask,
            shape=shape,
            material=material,
            color=color,
            nose_pads=material == FrameMaterial.METAL,
        )

    @staticmethod
    def _union_binary_masks(results, image_size: tuple[int, int]) -> np.ndarray | None:
        height, width = image_size
        combined = np.zeros((height, width), dtype=np.uint8)
        found = False
        for result in results:
            if result.masks is None:
                continue
            for mask_data in result.masks.data:
                mask = cv2.resize(
                    mask_data.cpu().numpy().astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)
                found = True
        return combined if found and combined.any() else None
