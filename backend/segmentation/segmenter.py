"""Eyewear segmentation with validated YOLO class contracts.

Supports either a single-class full-eyewear silhouette model or the agreed
six-class part model. Per-class masks are preserved; the combined silhouette
is only a compatibility/output mask and must not be used as a part mask.
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

_PART_ALIASES: dict[int, set[str]] = {
    0: {"eyewearrim", "rim", "frame", "eyewearfront"},
    1: {"eyeweartemple", "temple", "templearm", "arm", "legs"},
    2: {"bridge", "nosebridge"},
    3: {"leftlens", "lensleft"},
    4: {"rightlens", "lensright"},
    5: {"nosepad", "nosepads"},
}
_SILHOUETTE_ALIASES = {
    "eyewear", "glasses", "spectacles", "eyeglasses", "sunglasses",
    "frame", "eyewearmask", "eyewearoutline", "eyewear_silhouette",
}
_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".xml", ".mlpackage", ".tflite"}


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _as_numpy(value: Any) -> np.ndarray:
    """Convert numpy-like or torch-like values without requiring torch in tests."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _candidate_paths() -> list[Path]:
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
    """Find a fine-tuned eyewear checkpoint; generic COCO weights are never implicit."""
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
    """Accept a known one-class silhouette or the exact six-part class contract."""
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
        if name not in {_normalise_label(v) for v in _SILHOUETTE_ALIASES}:
            raise ValueError(
                f"Unsupported one-class checkpoint label {normalized[0]!r}; "
                "expected an eyewear/glasses/spectacles silhouette class."
            )
        return count, normalized, False

    if count != 6:
        raise ValueError(
            f"Incompatible eyewear model: expected 1 class (eyewear silhouette) or 6 named parts; found {count}."
        )

    for class_id, aliases in _PART_ALIASES.items():
        actual = _normalise_label(normalized[class_id])
        if actual not in {_normalise_label(v) for v in aliases}:
            raise ValueError(
                f"Wrong class contract at ID {class_id}: expected one of {sorted(aliases)}, "
                f"received {normalized[class_id]!r}. Update data.yaml/model together."
            )
    return count, normalized, True


class GlassesSegmenter:
    """Return a combined eyewear mask plus class-specific masks and diagnostics."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        require_model: bool | None = None,
        confidence_threshold: float | None = None,
    ):
        if require_model is None:
            require_model = os.environ.get("DEFIRM_REQUIRE_YOLO", "").strip().lower() in {
                "1", "true", "yes"
            }
        self.require_model = bool(require_model)

        if confidence_threshold is None:
            raw_conf = os.environ.get("DEFIRM_YOLO_CONF", "0.25").strip()
            try:
                confidence_threshold = float(raw_conf)
            except ValueError as exc:
                raise ValueError("DEFIRM_YOLO_CONF must be a number between 0 and 1.") from exc
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("YOLO confidence threshold must be between 0 and 1.")
        self.confidence_threshold = float(confidence_threshold)

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
            message = (
                "No fine-tuned eyewear checkpoint found. Set DEFIRM_YOLO_MODEL "
                "to a validated one-class or six-class eyewear segmentation model."
            )
            if self.require_model:
                raise RuntimeError(message + " DEFIRM_REQUIRE_YOLO is enabled; refusing OpenCV fallback.")
            logger.warning("%s Using low-confidence OpenCV fallback for development only.", message)
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
        """Return full-resolution binary masks and segmentation provenance."""
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Expected a non-empty BGR image with shape (height, width, 3).")
        image = np.ascontiguousarray(image)
        if self._model is not None:
            return self._segment_yolo(image)
        return self._segment_opencv(image)

    def _segment_yolo(self, image: np.ndarray) -> dict[str, Any]:
        h, w = image.shape[:2]
        class_masks = {
            class_id: np.zeros((h, w), dtype=np.uint8)
            for class_id in self.model_names
        }
        detections = 0

        results = self._model(image, verbose=False, conf=self.confidence_threshold)
        for result in results:
            result_masks = getattr(result, "masks", None)
            if result_masks is None:
                continue
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                raise ValueError("YOLO returned segmentation masks without class boxes; cannot assign part labels.")

            mask_data = _as_numpy(result_masks.data)
            if mask_data.ndim == 2:
                mask_data = mask_data[None, ...]
            if mask_data.ndim != 3:
                raise ValueError(f"Unexpected YOLO mask tensor shape: {mask_data.shape}")

            class_ids = _as_numpy(boxes.cls).reshape(-1)
            confidences = (
                _as_numpy(boxes.conf).reshape(-1)
                if getattr(boxes, "conf", None) is not None
                else np.ones(len(class_ids), dtype=np.float32)
            )
            if len(mask_data) != len(class_ids) or len(mask_data) != len(confidences):
                raise ValueError(
                    "YOLO returned a different number of masks, class IDs and confidence scores; "
                    "refusing to pair unrelated detections."
                )

            for item, raw_class_id, confidence in zip(mask_data, class_ids, confidences):
                if float(confidence) < self.confidence_threshold:
                    continue
                class_id = int(raw_class_id)
                if class_id not in class_masks:
                    raise ValueError(
                        f"YOLO predicted unknown class ID {class_id}; "
                        f"checkpoint declares {sorted(class_masks)}."
                    )
                resized = cv2.resize(
                    np.asarray(item, dtype=np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary = (resized >= 0.5).astype(np.uint8) * 255
                if not np.any(binary):
                    continue
                class_masks[class_id] = cv2.bitwise_or(class_masks[class_id], binary)
                detections += 1

        combined = np.zeros((h, w), dtype=np.uint8)
        part_masks: dict[str, np.ndarray] = {}
        for class_id, class_mask in class_masks.items():
            combined = cv2.bitwise_or(combined, class_mask)
            part_masks[self.model_names[class_id]] = class_mask

        return {
            "front": combined,
            # Compatibility aliases. A single uploaded photo is still one observed view.
            "side": combined,
            "full": combined,
            "part_masks": part_masks,
            "class_masks": class_masks,
            "model_type": self.model_type,
            "detections": detections,
            "class_count": self.model_class_count,
            "class_names": dict(self.model_names),
            "supports_part_masks": self.supports_part_masks,
            "observed_views": ["front"],
            "confidence_threshold": self.confidence_threshold,
        }

    def _segment_opencv(self, image: np.ndarray) -> dict[str, Any]:
        """Low-confidence fallback for development on high-contrast product photos."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < 55).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.zeros_like(gray)
            detections = 0
        else:
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
            detections = 1

        return {
            "front": mask,
            "side": mask,
            "full": mask,
            "part_masks": {"eyewear": mask},
            "class_masks": {},
            "model_type": self.model_type,
            "detections": detections,
            "class_count": 0,
            "class_names": {},
            "supports_part_masks": False,
            "observed_views": ["front"],
            "confidence_threshold": None,
        }
