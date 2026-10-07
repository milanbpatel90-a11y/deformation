"""Glasses segmentation using YOLOv8-Seg with OpenCV fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".xml", ".mlpackage", ".tflite"}
# Candidate model paths searched in order. Override with DEFIRM_YOLO_MODEL env var.
#
# models/best.pt is deliberately searched LAST. It currently holds a six-class
# part model (eyewear_rim, eyewear_temple, bridge, left_lens, right_lens,
# nose_pad) that returns zero detections on every image in test_images/. While
# it was listed first it silently shadowed the trained one-class `eyewear`
# weights under runs/segment/train/, so the pipeline received an empty mask and
# the measurement stage quietly substituted default sizes. The one-class weights
# are what this repo's own setup guide means by "cp
# runs/segment/train/weights/best.pt models/best.pt", and a single foreground
# class is what the measurement and view-classification stages expect.
_CANDIDATE_PATHS = [
    os.environ.get("DEFIRM_YOLO_MODEL"),
    _PROJECT_ROOT / "models" / "glasses_seg.pt",
    _PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt",
    _PROJECT_ROOT / "runs" / "segment" / "runs" / "segment" / "eyewear_seg" / "weights" / "best.pt",
    _PROJECT_ROOT / "models" / "best.pt",
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
    - Merges every detected class mask into one foreground mask, so a part model
      still yields the single "eyewear" region the measurement stages expect.
    - Falls back to OpenCV threshold/contour when no model is found, and also
      when a model is present but detects nothing in the frame.
    """

    #: Detections below this confidence are discarded before any merging. The
    #: model emits occasional low-confidence boxes on empty background, and on a
    #: rendered orbit one such box (confidence 0.35) stretched a 431 px frame
    #: into a 749 px mask.
    MIN_DETECTION_CONFIDENCE = 0.35
    #: Boxes overlapping the best detection at least this much are the same object.
    DETECTION_MERGE_IOU = 0.6
    #: A part of the same object sits within this fraction of the anchor's width...
    DETECTION_MERGE_GAP_RATIO = 0.3
    #: ...and is not dramatically wider than it. A coarse superset fails this.
    DETECTION_MAX_WIDTH_RATIO = 2.5

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self.model_type = "opencv_fallback"
        self.class_count = 0

        if YOLO is None:
            return  # ultralytics not installed

        resolved = Path(model_path).expanduser() if model_path else _find_model()
        if resolved and resolved.is_file():
            self._model = YOLO(str(resolved))
            self.model_type = f"yolo:{resolved.name}"
            names = getattr(self._model, "names", None) or {}
            self.class_count = len(names)
            if self.class_count > 1:
                # Every class mask is still merged into one foreground mask, so
                # a part model works -- it is just not what the measurement
                # stages are tuned against, and some of them detect nothing.
                LOGGER.warning(
                    "Segmentation model %s has %d classes (%s); this pipeline expects a "
                    "single foreground 'eyewear' class and merges all masks.",
                    resolved.name, self.class_count, sorted(names.values())[:6],
                )
        else:
            self.class_count = 0

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return binary masks + metadata for frame regions.

        Keys: front, side, full, model_type, detections, fallback
        """
        if self._model is not None:
            result = self._segment_yolo(image)
            if result["detections"] > 0:
                result["fallback"] = False
                return result
            # The model found nothing. A classical mask is strictly better than
            # an empty one, because downstream code cannot tell "no glasses
            # here" from "the model missed" and would substitute default sizes.
            classical = self._segment_opencv(image)
            if classical["detections"] > 0:
                classical["model_type"] = f"{self.model_type}+opencv_fallback"
                classical["fallback"] = True
                return classical
            result["fallback"] = False
            return result
        result = self._segment_opencv(image)
        result["fallback"] = False
        return result

    def _segment_yolo(self, image: np.ndarray) -> dict[str, np.ndarray]:
        results = self._model(image, verbose=False)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections = 0
        scores: list[float] = []
        boxes: list[tuple[float, float, float, float]] = []
        kept_masks: list[np.ndarray] = []

        for result in results:
            if result.masks is None:
                continue
            for index, m in enumerate(result.masks.data):
                confidence = 1.0
                if result.boxes is not None and index < len(result.boxes):
                    confidence = float(result.boxes.conf[index])
                if confidence < self.MIN_DETECTION_CONFIDENCE:
                    continue
                resized = cv2.resize(
                    m.cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary = (resized > 0.5).astype(np.uint8)
                rows = np.flatnonzero(binary.any(axis=1))
                cols = np.flatnonzero(binary.any(axis=0))
                if rows.size == 0 or cols.size == 0:
                    continue
                cols = np.flatnonzero(binary.any(axis=0))
                boxes.append(
                    (float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1]))
                )
                scores.append(confidence)
                kept_masks.append(binary)

        keep = self._consistent_detections(boxes, scores)
        for index in keep:
            mask = np.maximum(mask, kept_masks[index] * 255)
            detections += 1

        return {
            "front": mask,
            "side": mask,
            "full": mask,
            "model_type": self.model_type,
            "detections": detections,
            "detection_scores": [round(scores[i], 3) for i in keep],
        }

    def _consistent_detections(
        self,
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
    ) -> list[int]:
        """Keep the detections that plausibly describe one pair of glasses.

        The one-class model sometimes returns a second, coarser box that covers
        the real object *and* a swathe of empty background. Unioning every
        detection then inflates the silhouette -- measured on a rendered orbit,
        a 347 px frame became a 668 px mask at one pose and a 431 px frame became
        729 px at another, which corrupted the frontal reference width and made
        every downstream angle and scale wrong.

        Parts of one object (two rims, a bridge) are *near* each other, so a
        detection is merged when it overlaps the anchor strongly or sits within a
        short horizontal gap of it. A coarse superset is much wider than the
        anchor and reaches far beyond it, so it is dropped instead.
        """
        if not boxes:
            return []
        anchor = int(np.argmax(scores))
        ax0, _ay0, ax1, _ay1 = boxes[anchor]
        anchor_w = max(ax1 - ax0, 1.0)
        keep = [anchor]
        for index, box in enumerate(boxes):
            if index == anchor:
                continue
            bx0, _by0, bx1, _by1 = boxes[index]
            overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            union = max(ax1, bx1) - min(ax0, bx0)
            iou = overlap / union if union > 0 else 0.0
            gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            width = max(bx1 - bx0, 1.0)
            if iou >= self.DETECTION_MERGE_IOU:
                keep.append(index)
            elif gap <= self.DETECTION_MERGE_GAP_RATIO * anchor_w and width <= (
                self.DETECTION_MAX_WIDTH_RATIO * anchor_w
            ):
                # A nearby part of the same object.
                keep.append(index)
        return keep

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
