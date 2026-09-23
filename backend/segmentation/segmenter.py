"""YOLOv8-Seg eyewear segmenter with a strict 1-class model contract.

The production scope for this repository is binary segmentation:
    class 0 = eyewear

The segmenter deliberately does NOT fall back to a generic COCO YOLO
checkpoint. If no fine-tuned eyewear checkpoint is available, it uses the
existing OpenCV fallback and logs an ERROR. If a checkpoint is present but
does not match the 1-class contract, initialization raises RuntimeError so a
wrong model can never silently run in production.
"""

from __future__ import annotations

import logging
import os
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
_MODEL_EXTENSIONS = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".xml",
    ".mlpackage",
    ".tflite",
}

EXPECTED_CLASS_COUNT = 1
EXPECTED_CLASS_NAMES = {0: "eyewear"}

# Only fine-tuned repository/canonical checkpoints are candidates.
# A generic YOLOv8 COCO checkpoint is intentionally NOT included.
_CANDIDATE_PATHS = [
    os.environ.get("DEFIRM_YOLO_MODEL"),
    _PROJECT_ROOT / "models" / "glasses_seg.pt",
    _PROJECT_ROOT / "models" / "best.pt",
    _PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt",
]


def _normalise_class_names(raw_names: Any) -> dict[int, str]:
    """Convert Ultralytics class names to a deterministic int -> str mapping."""
    if isinstance(raw_names, dict):
        result: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result

    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}

    return {}


def _find_model() -> Path | None:
    """Return the first available fine-tuned checkpoint.

    Search order:
      1. DEFIRM_YOLO_MODEL override
      2. canonical models/glasses_seg.pt
      3. models/best.pt
      4. training output best.pt

    Generic pretrained YOLO weights are never considered here.
    """
    seen: set[str] = set()

    for candidate in _CANDIDATE_PATHS:
        if not candidate:
            continue

        path = Path(candidate).expanduser()
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)

        if key in seen:
            continue
        seen.add(key)

        if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS:
            return path

    return None


def _validate_model_contract(model: Any, model_path: Path) -> dict[int, str]:
    """Validate that an Ultralytics model is exactly the 1-class eyewear model."""
    raw_model = getattr(model, "model", None)
    raw_names = getattr(raw_model, "names", None)
    names = _normalise_class_names(raw_names)

    class_count = len(names)
    if class_count != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            "YOLO MODEL CONTRACT VIOLATION: "
            f"loaded '{model_path}' with {class_count} classes {names!r}. "
            f"This Deformation version requires exactly "
            f"{EXPECTED_CLASS_COUNT} class: {EXPECTED_CLASS_NAMES!r}. "
            "A generic COCO checkpoint such as yolov8n-seg.pt must not be used."
        )

    if names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            "YOLO MODEL CONTRACT VIOLATION: "
            f"loaded '{model_path}' with class names {names!r}; "
            f"expected exactly {EXPECTED_CLASS_NAMES!r}."
        )

    return names


class GlassesSegmenter:
    """Segment eyewear from product images using the strict 1-class model."""

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self.model_type = "opencv_fallback"
        self.model_path: Path | None = None
        self.class_names: dict[int, str] = {}

        if YOLO is None:
            logger.error(
                "Ultralytics is not installed; using OpenCV fallback. "
                "Install the project requirements to enable YOLOv8-Seg."
            )
            return

        resolved = Path(model_path).expanduser() if model_path else _find_model()

        if resolved is None:
            logger.error(
                "Fine-tuned YOLOv8-Seg model not found. Checked: %s. "
                "No generic pretrained checkpoint will be used; "
                "falling back to OpenCV segmentation.",
                ", ".join(str(path) for path in _CANDIDATE_PATHS if path),
            )
            return

        if not resolved.is_file():
            logger.error(
                "Configured YOLOv8-Seg model does not exist: %s. "
                "Falling back to OpenCV segmentation.",
                resolved,
            )
            return

        try:
            loaded_model = YOLO(str(resolved))
            class_names = _validate_model_contract(loaded_model, resolved)
        except Exception as exc:
            # A present-but-invalid model is materially different from a missing
            # model. Do not silently degrade to OpenCV because that would hide a
            # deployment/configuration error.
            logger.exception(
                "Failed to load/validate YOLOv8-Seg model '%s'.",
                resolved,
            )
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(
                f"Unable to load the configured YOLOv8-Seg model: {resolved}"
            ) from exc

        self._model = loaded_model
        self.model_path = resolved
        self.class_names = class_names
        self.model_type = f"yolo:{resolved.name}"

        logger.info(
            "Loaded fine-tuned YOLOv8-Seg model: %s | classes=%s",
            resolved,
            self.class_names,
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def segment(self, image: np.ndarray) -> dict[str, Any]:
        """Return a union mask and per-class masks."""
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

            for index, mask_data in enumerate(result.masks.data):
                resized = cv2.resize(
                    mask_data.detach().cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary = (resized > 0.5).astype(np.uint8) * 255

                class_id = (
                    int(boxes_cls[index])
                    if boxes_cls is not None and index < len(boxes_cls)
                    else 0
                )

                class_masks[class_id] = np.maximum(
                    class_masks.get(
                        class_id,
                        np.zeros((h, w), dtype=np.uint8),
                    ),
                    binary,
                )

                class_name = self.class_names[class_id]
                components[class_name] = np.maximum(
                    components.get(
                        class_name,
                        np.zeros((h, w), dtype=np.uint8),
                    ),
                    binary,
                )
                detections += 1

        union = np.zeros((h, w), dtype=np.uint8)
        for component_mask in class_masks.values():
            union = np.maximum(union, component_mask)

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
        contours, _ = cv2.findContours(
            dark,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

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
            normalized_distance = (
                ((center[0] - image_center[0]) / max(w, 1)) ** 2
                + ((center[1] - image_center[1]) / max(h, 1)) ** 2
            )
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
