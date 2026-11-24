"""Produce and persist the saved assets for one selected reference frame:
original image, per-component masks, a transparent cutout, and a visual
overlay for human review. Never invents a mask for a component the model
didn't actually detect -- callers see `missing_components` on the
`DetectionResult` instead of a fabricated blank mask standing in for one.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.video_extraction.glasses_detector import EXPECTED_COMPONENTS, GlassesFrameDetector
from backend.video_extraction.schemas import DetectionResult, SelectedFrame, ViewLabel
from backend.video_extraction.storage import VideoExtractionStorage

_OVERLAY_COLOR = (36, 231, 139)  # BGR, drawn at partial alpha over the original
_OVERLAY_ALPHA = 0.4


def _encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()


def build_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[:, :] = _OVERLAY_COLOR
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(frame, 1 - _OVERLAY_ALPHA, colored, _OVERLAY_ALPHA, 0)[mask_bool]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, _OVERLAY_COLOR, 2)
    return overlay


def build_cutout(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGBA cutout: original pixels where the mask is set, transparent elsewhere."""
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def persist_selected_frame(
    storage: VideoExtractionStorage,
    job_id: str,
    view: ViewLabel,
    index_in_view: int,
    frame: np.ndarray,
    detection: DetectionResult,
    combined_mask: np.ndarray,
    per_class_masks: dict[str, np.ndarray],
    frame_index: int,
    timestamp_sec: float,
    view_confidence: float,
    composite_score: float,
    duplicate_score: int = 0,
    is_additional: bool = False,
    manually_replaced: bool = False,
) -> SelectedFrame:
    """Encode and save every asset for one selected frame and return the
    `SelectedFrame` record (with storage-relative paths, not local
    filesystem paths -- those never need to reach the client).

    Folder layout matches the task brief's structure: ``selected/`` for the
    chosen reference images, ``crops/`` for the transparent eyewear-only
    cutout, ``masks/`` and ``overlays/`` as before. 1-indexed, zero-padded
    filenames (``front_01.png``) per the brief's example tree.
    """
    base = f"{view.value}_{index_in_view + 1:02d}"

    image_rel = f"selected/{base}.png"
    storage.save(job_id, image_rel, _encode_png(frame))

    overlay_rel = f"overlays/{base}.png"
    storage.save(job_id, overlay_rel, _encode_png(build_overlay(frame, combined_mask)))

    crop_rel = f"crops/{base}.png"
    storage.save(job_id, crop_rel, _encode_png(build_cutout(frame, combined_mask)))

    mask_paths: dict[str, str] = {}
    for class_name, class_mask in per_class_masks.items():
        rel = f"masks/{base}_{class_name}.png"
        storage.save(job_id, rel, _encode_png(class_mask))
        mask_paths[class_name] = rel
    if not per_class_masks:
        rel = f"masks/{base}_combined.png"
        storage.save(job_id, rel, _encode_png(combined_mask))
        mask_paths["combined"] = rel

    component_visibility = {name: name in per_class_masks for name in EXPECTED_COMPONENTS}

    return SelectedFrame(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        view=view,
        view_confidence=round(view_confidence, 3),
        composite_score=composite_score,
        image_path=image_rel,
        overlay_path=overlay_rel,
        cutout_path=crop_rel,
        mask_paths=mask_paths,
        detection=detection,
        duplicate_score=duplicate_score,
        is_additional=is_additional,
        manually_replaced=manually_replaced,
        component_visibility=component_visibility,
    )
