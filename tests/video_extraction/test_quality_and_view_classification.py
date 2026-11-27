"""Unit tests for quality_analyzer and view_classifier.

These were previously untested in isolation (only exercised indirectly,
and only on a solid-color synthetic clip that produces zero detections,
through test_api.py's end-to-end flow). Pure-function unit coverage here
closes that gap for the brief's test topics "blur detection", "exposure
and contrast", and "view classification".
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.video_extraction.quality_analyzer import analyze_frame, compute_exposure, compute_sharpness
from backend.video_extraction.schemas import ViewLabel
from backend.video_extraction.view_classifier import classify_frame


# --------------------------------------------------------------------------
# Sharpness / blur
# --------------------------------------------------------------------------

def _checkerboard(size=256) -> np.ndarray:
    gray = np.zeros((size, size), dtype=np.uint8)
    step = 8
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                gray[y : y + step, x : x + step] = 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_blurry_frame_scores_lower_sharpness_than_the_sharp_source():
    sharp = _checkerboard()
    blurry = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert compute_sharpness(blurry) < compute_sharpness(sharp)


def test_analyze_frame_flags_blur_below_threshold():
    from backend.video_extraction.config import get_settings

    settings = get_settings()
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)  # no edges at all: minimal Laplacian variance
    metrics = analyze_frame(flat, settings=settings)
    assert metrics.is_blurry
    assert any("blur" in r.lower() or "sharp" in r.lower() for r in metrics.reasons)


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------

def test_compute_exposure_detects_near_black_frame_as_dark():
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    dark_fraction, bright_fraction = compute_exposure(black)
    assert dark_fraction > 0.9
    assert bright_fraction < 0.1


def test_compute_exposure_detects_near_white_frame_as_bright():
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    dark_fraction, bright_fraction = compute_exposure(white)
    assert bright_fraction > 0.9
    assert dark_fraction < 0.1


def test_analyze_frame_rejects_overexposed_frame_on_exposure_not_sharpness():
    # Fully blown-out highlights: bright_fraction far above the configured
    # max (0.6 by default), regardless of any texture/sharpness present.
    overexposed = np.full((100, 100, 3), 250, dtype=np.uint8)
    metrics = analyze_frame(overexposed)
    assert not metrics.exposure_ok
    assert any("overexposed" in r.lower() for r in metrics.reasons)


# --------------------------------------------------------------------------
# View classification (pure mask-geometry heuristic -- see module docstring)
# --------------------------------------------------------------------------

def _rect_mask(frame_size: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_size
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = box
    mask[y : y + bh, x : x + bw] = 255
    return mask


def test_classify_frame_empty_mask_is_unknown():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    label, confidence = classify_frame(frame, mask)
    assert label == ViewLabel.UNKNOWN
    assert confidence == 0.0


def test_classify_frame_wide_symmetric_box_is_front():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Wide (aspect ~2.5), perfectly left/right symmetric -> front.
    mask = _rect_mask((200, 300), (50, 80, 200, 80))
    label, confidence = classify_frame(frame, mask)
    assert label == ViewLabel.FRONT
    assert confidence > 0.5


def test_classify_frame_very_elongated_box_is_a_side_view():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    # Very elongated (aspect > 4.3) -> a side profile, not front/45.
    mask = _rect_mask((200, 400), (10, 90, 380, 20))
    label, confidence = classify_frame(frame, mask)
    assert label in (ViewLabel.LEFT_SIDE, ViewLabel.RIGHT_SIDE)
    assert confidence > 0.0


def test_classify_frame_squat_box_is_top_not_front():
    frame = np.zeros((300, 200, 3), dtype=np.uint8)
    # Taller-than-wide (aspect < 0.95) -> top, never fabricated as front.
    mask = _rect_mask((300, 200), (60, 20, 80, 260))
    label, confidence = classify_frame(frame, mask)
    assert label == ViewLabel.TOP
    assert confidence > 0.0


def test_classify_frame_squat_box_with_dominant_nose_pad_evidence_is_bottom():
    frame = np.zeros((300, 200, 3), dtype=np.uint8)
    mask = _rect_mask((300, 200), (60, 20, 80, 260))
    rim_mask = _rect_mask((300, 200), (60, 20, 80, 260))
    # Nose-pad area well above the configured ratio of the rim area -> bottom,
    # per the module's documented, narrow, evidence-gated rule (never fired
    # from bbox shape alone).
    nose_pad_mask = _rect_mask((300, 200), (70, 100, 60, 150))
    label, confidence = classify_frame(frame, mask, {"nose_pad": nose_pad_mask, "eyewear_rim": rim_mask})
    assert label == ViewLabel.BOTTOM
    assert confidence > 0.0


def test_classify_frame_never_emits_bottom_without_component_evidence():
    """The narrow bottom-detection rule must never fire from bbox shape
    alone -- only `test_..._bottom` above (with explicit nose-pad/rim
    per-class masks) should ever produce BOTTOM; the same squat silhouette
    with no per_class_masks argument must fall back to TOP."""
    frame = np.zeros((300, 200, 3), dtype=np.uint8)
    mask = _rect_mask((300, 200), (60, 20, 80, 260))
    label, _confidence = classify_frame(frame, mask, per_class_masks=None)
    assert label == ViewLabel.TOP


def test_classify_frame_moderately_asymmetric_elongated_box_is_a_45_degree_view():
    frame = np.zeros((200, 350, 3), dtype=np.uint8)
    mask = np.zeros((200, 350), dtype=np.uint8)
    # One connected wedge-shaped blob (classify_frame's contour step picks
    # the single largest contour, so two disjoint rectangles would not
    # produce the combined bbox this test wants): a short, thin stub on the
    # left (rows 90:110) touching a tall, full block on the right (rows
    # 60:140), sharing the column-150 boundary so cv2.findContours sees one
    # component. Overall bbox = x:50..320 (width 270, aspect 3.375, in the
    # 45-degree band); left half (50..185) is mostly empty while right half
    # (185..320) is fully filled -> clearly asymmetric (symmetry well below
    # the front threshold of 0.72) without tipping into a true side profile.
    mask[90:110, 50:150] = 255   # left stub
    mask[60:140, 150:320] = 255  # right block (full height)
    label, confidence = classify_frame(frame, mask)
    assert label == ViewLabel.FRONT_RIGHT_45
    assert confidence > 0.0
