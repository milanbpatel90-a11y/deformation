"""Glasses segmentation using YOLOv8-Seg.

The segmenter exposes both:
- front: union mask kept for backward compatibility with the existing
  measurement/classification pipeline.
- components: per-class masks for the component-aware pipeline.

Model resolution is deterministic and prefers a repository fine-tuned
checkpoint over generic pretrained YOLO weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".xml", ".mlpackage", ".tflite"}

# Fine-tuned checkpoints must precede the generic pretrained checkpoint.
_CANDIDATE_PATHS = [
    os.environ.get("DEFIRM_YOLO_MODEL"),
    _PROJECT_ROOT / "models" / "best.pt",
    _PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt",
    _PROJECT_ROOT / "models" / "glasses_seg.pt",
    _PROJECT_ROOT / "yolov8n-seg.pt",
]


def _find_model() -> Path | None:
    """Resolve the best available checkpoint in deterministic priority order."""
    seen: set[str] = set()
    for candidate in _CANDIDATE_PATHS:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS:
            return path
    return None


class GlassesSegmenter:
    """Run YOLOv8-Seg and expose backwards-compatible + component masks."""

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self.model_type = "opencv_fallback"
        self.model_path: Path | None = None
        self.class_names: dict[int, str] = {}

        if YOLO is None:
            return

        resolved = Path(model_path).expanduser() if model_path else _find_model()
        if resolved and resolved.is_file():
            self._model = YOLO(str(resolved))
            self.model_path = resolved
            self.model_type = f"yolo:{resolved.name}"

            raw_names: Any = getattr(getattr(self._model, "model", None), "names", {})
            if isinstance(raw_names, dict):
                self.class_names = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, (list, tuple)):
                self.class_names = {i: str(v) for i, v in enumerate(raw_names)}

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def segment(self, image: np.ndarray) -> dict[str, Any]:
        """Return a union mask and per-class component masks."""
        if self._model is not None:
            return self._segment_yolo(image)
        return self._segment_opencv(image)

    def _segment_yolo(self, image: np.ndarray) -> dict[str, Any]:
        results = self._model(image, verbose=False)
        h, w = image.shape[:2]
        components: dict[str, np.ndarray] = {}
        class_masks: dict[int, np.ndarray] = {}
        detections = 0

        for result in results:
            if result.masks is None:
                continue

            boxes_cls = None
            if result.boxes is not None and result.boxes.cls is not None:
                boxes_cls = result.boxes.cls.detach().cpu().numpy().astype(int)

            for i, m in enumerate(result.masks.data):
                resized = cv2.resize(
                    m.detach().cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary = (resized > 0.5).astype(np.uint8) * 255
                class_id = int(boxes_cls[i]) if boxes_cls is not None and i < len(boxes_cls) else 0

                class_masks[class_id] = np.maximum(
                    class_masks.get(class_id, np.zeros((h, w), dtype=np.uint8)),
                    binary,
                )

                class_name = self.class_names.get(class_id, str(class_id))
                components[class_name] = np.maximum(
                    components.get(class_name, np.zeros((h, w), dtype=np.uint8)),
                    binary,
                )
                detections += 1

        union = np.zeros((h, w), dtype=np.uint8)
        for mask in class_masks.values():
            union = np.maximum(union, mask)

        return {
            "front": union,
            "side": union,
            "full": union,
            "class_masks": class_masks,
            "components": components,
            "class_names": dict(self.class_names),
            "model_type": self.model_type,
            "model_path": str(self.model_path) if self.model_path else None,
            "num_classes": self.num_classes,
            "detections": detections,
        }

    def _segment_opencv(self, image: np.ndarray) -> dict[str, Any]:
        """Fallback: dark-frame threshold with center-biased contour selection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        dark = (gray < 55).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            empty = np.zeros_like(gray)
            return {
                "front": empty,
                "side": empty,
                "full": empty,
                "class_masks": {},
                "components": {},
                "class_names": {},
                "model_type": self.model_type,
                "model_path": None,
                "num_classes": 0,
                "detections": 0,
            }

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

        return {
            "front": mask,
            "side": mask,
            "full": mask,
            "class_masks": {},
            "components": {},
            "class_names": {},
            "model_type": self.model_type,
            "model_path": None,
            "num_classes": 0,
            "detections": 1,
        }
