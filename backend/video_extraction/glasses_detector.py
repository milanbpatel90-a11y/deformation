"""Per-frame glasses detection/segmentation for the video pipeline.

Reuses `backend.segmentation.segmenter.GlassesSegmenter` (the same class
and, within one pipeline run, the same loaded model instance used by the
image-based `DeformationPipeline`) instead of loading YOLO a second time
or replacing it. `GlassesSegmenter.segment()` only returns one merged
binary mask; this module additionally reaches into the already-loaded
`ultralytics` model (when present) to recover per-class masks/confidences
for the richer per-component reporting the brief asks for. When the
segmenter is running in its OpenCV fallback mode (no YOLO weights found),
only the combined mask/bbox is available and every per-class field is
reported empty/zero -- this is surfaced, not invented.

Verified real class set baked into `models/best.pt` via
``YOLO('models/best.pt').names`` in this environment:
``{0: 'eyewear_rim', 1: 'eyewear_temple', 2: 'bridge', 3: 'left_lens',
4: 'right_lens', 5: 'nose_pad'}``. Note there is one merged
``eyewear_temple`` class, not separate left/right temple classes, and one
merged ``eyewear_rim`` class -- left/right labeling for rim/temple is not
provided by the model and is never fabricated here.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from backend.segmentation.segmenter import GlassesSegmenter
from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.schemas import DetectionResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_default_segmenter() -> GlassesSegmenter:
    """Build the `GlassesSegmenter` this feature should use by default.

    `GlassesSegmenter`'s own built-in candidate-path list (see
    backend/segmentation/segmenter.py) checks `DEFIRM_YOLO_MODEL`, then
    `models/glasses_seg.pt`, then the repo-root `yolov8n-seg.pt` (a
    generic, non-glasses-specific COCO-pretrained model), then a
    `runs/segment/train/weights/best.pt`. It does NOT look at
    `models/best.pt` -- the actual trained glasses-component model
    verified in this repo (6 classes: eyewear_rim, eyewear_temple, bridge,
    left_lens, right_lens, nose_pad). Whatever the existing `/api/deform*`
    endpoints resolve to is left completely untouched (this does not
    modify segmenter.py or its candidate list). For the new video
    extraction feature specifically -- whose whole point is per-component
    masks/confidences -- we pass `models/best.pt` explicitly when present
    and `DEFIRM_YOLO_MODEL` is not set, so the feature doesn't silently
    end up running on a generic COCO segmentation model.
    """
    explicit = os.environ.get("DEFIRM_YOLO_MODEL")
    if explicit:
        return GlassesSegmenter(model_path=explicit)
    best_pt = _PROJECT_ROOT / "models" / "best.pt"
    if best_pt.is_file():
        return GlassesSegmenter(model_path=best_pt)
    return GlassesSegmenter()

# The component set a complete detection would ideally cover. Used only to
# report what's *missing* -- never to fabricate a mask for a class the
# model didn't actually produce.
EXPECTED_COMPONENTS = ("eyewear_rim", "eyewear_temple", "bridge", "left_lens", "right_lens", "nose_pad")


class GlassesFrameDetector:
    """Wraps one shared `GlassesSegmenter` instance for per-frame detection."""

    def __init__(self, segmenter: GlassesSegmenter | None = None):
        self.segmenter = segmenter or build_default_segmenter()

    def detect(self, frame: np.ndarray, settings: VideoExtractionSettings | None = None) -> tuple[DetectionResult, np.ndarray, dict[str, np.ndarray]]:
        """Return (DetectionResult, combined_mask, per_class_masks).

        per_class_masks is empty when running in the OpenCV fallback mode.
        """
        settings = settings or get_settings()
        h, w = frame.shape[:2]
        frame_area = float(h * w) or 1.0

        model = getattr(self.segmenter, "_model", None)
        if model is not None:
            return self._detect_yolo(frame, model, frame_area, settings)
        return self._detect_fallback(frame, frame_area, settings)

    def _detect_yolo(self, frame, model, frame_area, settings) -> tuple[DetectionResult, np.ndarray, dict[str, np.ndarray]]:
        h, w = frame.shape[:2]
        results = model(frame, verbose=False)
        combined = np.zeros((h, w), dtype=np.uint8)
        per_class_masks: dict[str, np.ndarray] = {}
        class_confidences: dict[str, float] = {}
        names = getattr(model, "names", {}) or {}

        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy().astype(float)
            for mask_tensor, cls_idx, conf in zip(result.masks.data, classes, confs):
                resized = cv2.resize(
                    mask_tensor.cpu().numpy().astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
                )
                binary = (resized > 0.5).astype(np.uint8) * 255
                combined = np.maximum(combined, binary)
                class_name = str(names.get(cls_idx, f"class_{cls_idx}"))
                if class_name in per_class_masks:
                    per_class_masks[class_name] = np.maximum(per_class_masks[class_name], binary)
                    class_confidences[class_name] = max(class_confidences[class_name], float(conf))
                else:
                    per_class_masks[class_name] = binary
                    class_confidences[class_name] = float(conf)

        detected = bool(per_class_masks)
        area_fraction = float(np.count_nonzero(combined)) / frame_area
        bbox = self._mask_bbox(combined)
        missing = [c for c in EXPECTED_COMPONENTS if c not in per_class_masks]
        confidence = float(np.mean(list(class_confidences.values()))) if class_confidences else 0.0

        result = DetectionResult(
            detected=detected and settings.min_glasses_area_fraction <= area_fraction <= settings.max_glasses_area_fraction,
            confidence=round(confidence, 4),
            bbox=bbox,
            area_fraction=round(area_fraction, 5),
            class_confidences={k: round(v, 4) for k, v in class_confidences.items()},
            missing_components=missing,
            model_type=self.segmenter.model_type,
        )
        return result, combined, per_class_masks

    def _detect_fallback(self, frame, frame_area, settings) -> tuple[DetectionResult, np.ndarray, dict[str, np.ndarray]]:
        masks = self.segmenter.segment(frame)
        combined = masks["front"]
        area_fraction = float(np.count_nonzero(combined)) / frame_area
        bbox = self._mask_bbox(combined)
        detected = masks.get("detections", 0) > 0
        result = DetectionResult(
            detected=detected and settings.min_glasses_area_fraction <= area_fraction <= settings.max_glasses_area_fraction,
            confidence=0.5 if detected else 0.0,  # OpenCV fallback has no real confidence signal
            bbox=bbox,
            area_fraction=round(area_fraction, 5),
            class_confidences={},
            missing_components=list(EXPECTED_COMPONENTS),  # fallback never produces per-class masks
            model_type=masks.get("model_type", "opencv_fallback"),
        )
        return result, combined, {}

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if not np.any(mask):
            return None
        ys, xs = np.where(mask > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1
