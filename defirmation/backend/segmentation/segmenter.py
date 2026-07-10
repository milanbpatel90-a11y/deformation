"""Glasses segmentation using YOLOv8-Seg with OpenCV fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]


class GlassesSegmenter:
    """Segment glasses regions from product images."""

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        if model_path and YOLO is not None:
            self._model = YOLO(str(model_path))

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return binary masks for frame regions.

        Keys: front, side, full
        """
        if self._model is not None:
            return self._segment_yolo(image)
        return self._segment_opencv(image)

    def _segment_yolo(self, image: np.ndarray) -> dict[str, np.ndarray]:
        results = self._model(image, verbose=False)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:
            if result.masks is None:
                continue
            for m in result.masks.data:
                resized = cv2.resize(
                    m.cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = np.maximum(mask, (resized > 0.5).astype(np.uint8) * 255)

        return {"front": mask, "side": mask, "full": mask}

    def _segment_opencv(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Fallback: dark-frame threshold with center-biased contour selection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        dark = (gray < 55).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"front": np.zeros_like(gray), "side": np.zeros_like(gray), "full": np.zeros_like(gray)}

        image_center = (w / 2.0, h / 2.0)

        def score(contour: np.ndarray) -> tuple[float, float, int]:
            x, y, contour_w, contour_h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            center = (x + contour_w / 2.0, y + contour_h / 2.0)
            normalized_distance = ((center[0] - image_center[0]) / max(w, 1)) ** 2
            normalized_distance += ((center[1] - image_center[1]) / max(h, 1)) ** 2
            aspect = contour_w / max(contour_h, 1)
            aspect_penalty = abs(aspect - 3.0)
            return normalized_distance, aspect_penalty, -area

        selected = min(contours, key=score)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [selected], -1, 255, -1)

        return {"front": mask, "side": mask, "full": mask}
