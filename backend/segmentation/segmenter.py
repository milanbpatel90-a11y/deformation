"""Eyewear segmentation with explicit, validated checkpoint selection.

Production rule: never silently load the generic COCO YOLO checkpoint as an
eyewear model. A custom one-class eyewear model is supported for the primary
deformation pipeline; the six-class model is required by the legacy part-mask
measurement route.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Class order shared with MaskToMeasurements and the six-class training format.
_PART_ALIASES: dict[int, set[str]] = {
    0: {"eyewearrim", "rim", "frame", "eyewearfront"},
    1: {"eyeweartemple", "temple", "templearm", "arm"},
    2: {"bridge", "nosebridge"},
    3: {"leftlens", "lensleft"},
    4: {"rightlens", "lensright"},
    5: {"nosepad", "nosepads"},
}
_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".xml", ".mlpackage", ".tflite"}


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _candidate_paths() -> list[Path]:
    """Resolve fine-tuned checkpoints, with a configured override first."""
    candidates: list[Path] = []
    env_model = os.environ.get("DEFIRM_YOLO_MODEL", "").strip()
    if env_model:
        candidates.append(Path(env_model).expanduser())
    candidates.extend([
        _PROJECT_ROOT / "models" / "glasses_seg.pt",
        _PROJECT_ROOT / "models" / "best.pt",
        _PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt",
        _PROJECT_ROOT / "runs" / "segment" / "runs" / "segment" / "eyewear_seg" / "weights" / "best.pt",
    ])
    return candidates


def _find_model() -> Path | None:
    """Find a fine-tuned eyewear model; generic yolov8n-seg.pt is never implicit."""
    env_model = os.environ.get("DEFIRM_YOLO_MODEL", "").strip()
    if env_model:
        configured = Path(env_model).expanduser()
        if not configured.is_file():
            raise FileNotFoundError(f"DEFIRM_YOLO_MODEL points to a missing file: {configured}")
        if configured.suffix.lower() not in _MODEL_EXTENSIONS:
            raise ValueError(f"Unsupported YOLO model file type: {configured.suffix}")
        return configured.resolve()

    for path in _candidate_paths():
        if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS:
            return path.resolve()
    return None


def _validate_model_names(names: Any) -> tuple[int, dict[int, str], bool]:
    """Accept either one eyewear-silhouette class or the agreed six-part map."""
    if isinstance(names, dict):
        normalized = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        normalized = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError("The eyewear checkpoint does not expose a readable class-name map.")

    if not normalized:
        raise ValueError("The eyewear checkpoint has no classes.")

    count = len(normalized)
    if set(normalized) != set(range(count)):
        raise ValueError(f"Model class IDs must be contiguous from zero; received {sorted(normalized)}.")

    if count == 1:
        name = _normalise_label(normalized[0])
        if name in {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}:
            raise ValueError(f"Checkpoint is not an eyewear model (class 0={normalized[0]!r}).")
        return count, normalized, False

    if count != 6:
        raise ValueError(
            f"Incompatible eyewear model: expected 1 class (eyewear silhouette) or 6 named parts; found {count}."
        )

    for class_id, aliases in _PART_ALIASES.items():
        actual = _normalise_label(normalized[class_id])
        if actual not in aliases:
            raise ValueError(
                f"Wrong class contract at ID {class_id}: expected one of {sorted(aliases)}, "
                f"received {normalized[class_id]!r}. Update data.yaml/model together."
            )
    return count, normalized, True


class GlassesSegmenter:
    """Segment the glasses silhouette using a validated model or explicit fallback."""

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self.model_type = "opencv_fallback"
        self.model_names: dict[int, str] = {}
        self.model_class_count = 0
        self.supports_part_masks = False

        configured = model_path or os.environ.get("DEFIRM_YOLO_MODEL")
        if configured and not Path(configured).expanduser().is_file():
            raise FileNotFoundError(f"Configured eyewear checkpoint not found: {configured}")

        resolved = Path(configured).expanduser() if configured else _find_model()
        if resolved is None:
            logger.warning(
                "No fine-tuned eyewear checkpoint found. Using OpenCV fallback; "
                "production generation should set DEFIRM_YOLO_MODEL to a validated model."
            )
            return
        if resolved.suffix.lower() not in _MODEL_EXTENSIONS:
            raise ValueError(f"Unsupported model extension: {resolved.suffix}")
        if YOLO is None:
            raise RuntimeError(
                f"Checkpoint configured at {resolved}, but ultralytics is not installed. "
                "Install the project requirements rather than silently switching segmentation methods."
            )

        loaded = YOLO(str(resolved))
        count, names, supports_parts = _validate_model_names(getattr(loaded, "names", None))
        self._model = loaded
        self.model_type = f"yolo:{resolved.name}"
        self.model_names = names
        self.model_class_count = count
        self.supports_part_masks = supports_parts
        logger.info(
            "Loaded validated eyewear model %s (classes=%s; part_masks=%s)",
            resolved, names, supports_parts
        )

    def segment(self, image: np.ndarray) -> dict[str, Any]:
        """Return a full eyewear mask and model diagnostics."""
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with shape (height, width, 3).")
        if self._model is not None:
            return self._segment_yolo(image)
        return self._segment_opencv(image)

    def _segment_yolo(self, image: np.ndarray) -> dict[str, Any]:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections = 0
        for result in self._model(image, verbose=False):
            if result.masks is None:
                continue
            for item in result.masks.data:
                resized = cv2.resize(
                    item.cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = np.maximum(mask, (resized > 0.5).astype(np.uint8) * 255)
                detections += 1
        return {
            "front": mask,
            # Compatibility aliases only: one photo is not three observed views.
            "side": mask,
            "full": mask,
            "model_type": self.model_type,
            "detections": detections,
            "class_count": self.model_class_count,
            "supports_part_masks": self.supports_part_masks,
        }

    def _segment_opencv(self, image: np.ndarray) -> dict[str, Any]:
        """Low-confidence fallback for high-contrast product photographs."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < 55).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.zeros_like(gray)
            return {
                "front": mask, "side": mask, "full": mask,
                "model_type": self.model_type, "detections": 0,
                "class_count": 0, "supports_part_masks": False,
            }

        h, w = gray.shape
        image_center = (w / 2.0, h / 2.0)

        def score(contour: np.ndarray) -> tuple[float, float, float]:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            center = (x + cw / 2.0, y + ch / 2.0)
            distance = ((center[0] - image_center[0]) / max(w, 1)) ** 2
            distance += ((center[1] - image_center[1]) / max(h, 1)) ** 2
            aspect_penalty = abs(cw / max(ch, 1) - 3.0)
            return distance, aspect_penalty, -area

        selected = min(contours, key=score)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [selected], -1, 255, -1)
        return {
            "front": mask, "side": mask, "full": mask,
            "model_type": self.model_type, "detections": 1,
            "class_count": 0, "supports_part_masks": False,
        }
