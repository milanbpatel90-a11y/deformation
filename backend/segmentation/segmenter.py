"""Glasses segmentation using YOLOv8-Seg with OpenCV fallback."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

# Candidate model paths searched in order. Override with DEFIRM_YOLO_MODEL env var.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".xml", ".mlpackage", ".tflite"}
_CANDIDATE_PATHS = [
    os.environ.get("DEFIRM_YOLO_MODEL"),
    _PROJECT_ROOT / "models" / "glasses_seg.pt",
    _PROJECT_ROOT / "models" / "best.pt",
    _PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt",
]


def _find_model() -> Path | None:
    for candidate in _CANDIDATE_PATHS:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS:
            return path
    return None


class GlassesSegmenter:
    """
    Segment glasses regions from product images.

    Stage: YOLO
    - Loads YOLOv8-seg model from DEFIRM_YOLO_MODEL env var or standard paths.
    - Falls back to OpenCV threshold/contour when no model is found.
    """

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self.model_type = "opencv_fallback"

        if YOLO is None:
            return  # ultralytics not installed

        resolved = Path(model_path).expanduser() if model_path else _find_model()
        if resolved and resolved.is_file():
            self._model = YOLO(str(resolved))
            self.model_type = f"yolo:{resolved.name}"

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return binary masks + metadata for frame regions.

        Keys: front, side, full, model_type
        """
        if self._model is not None:
            return self._segment_yolo(image)
        return self._segment_opencv(image)

    def _segment_yolo(self, image: np.ndarray) -> dict[str, np.ndarray]:
        results = self._model(image, verbose=False)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections = 0

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
                detections += 1

        return {
            "front": mask,
            "side": mask,
            "full": mask,
            "model_type": self.model_type,
            "detections": detections,
        }

    def _segment_opencv(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Fallback: dark-frame threshold with center-biased contour selection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        dark = (gray < 55).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            empty = np.zeros_like(gray)
            return {"front": empty, "side": empty, "full": empty,
                    "model_type": self.model_type, "detections": 0}

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

        # Keep the separate pieces of the frame together. Product photos often
        # break the dark threshold mask into two rims, a bridge and temples;
        # selecting just one "best" contour fed a single lens/hinge to view
        # classification and made the multi-view pipeline choose the wrong
        # image as its front view.
        candidates = []
        for contour in contours:
            x, y, contour_w, contour_h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            center_x = x + contour_w / 2.0
            center_y = y + contour_h / 2.0
            if (area >= max(2.0, h * w * 0.000002)
                    and w * 0.08 <= center_x <= w * 0.92
                    and h * 0.08 <= center_y <= h * 0.92):
                candidates.append(contour)

        if not candidates:
            candidates = [min(contours, key=score)]
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, candidates, -1, 255, -1)
        # Bridge small threshold gaps while keeping the image background out.
        close_size = max(3, min(11, int(round(min(h, w) * 0.012)) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return {
            "front": mask,
            "side": mask,
            "full": mask,
            "model_type": self.model_type,
            "detections": len(candidates),
        }
