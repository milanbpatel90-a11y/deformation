"""Heuristic viewing-angle classification for video frames.

IMPORTANT, stated plainly (see also the package README): this is a
heuristic classifier built from mask geometry (bounding-box aspect ratio,
left/right mask-area symmetry) layered on top of the YOLO segmentation.
It is NOT a calibrated pose estimator and does not know the true camera
rotation in degrees. "front_left_45" / "front_right_45" / "side_left" /
"side_right" are approximate buckets chosen from aspect-ratio and
asymmetry thresholds, not a geometric reconstruction of viewing angle.
When confidence is low the frame is labeled `unknown` rather than guessed,
and the dashboard lets a human confirm/correct it before approval.

This is distinct from `backend.fusion.view_classifier.ViewClassifier`,
which classifies single uploaded product photos into a coarser
front/side/top/perspective set for the existing multi-image pipeline.
This module classifies frames sampled from a *video*, over time, into the
finer-grained bucket set the video-extraction brief asks for
(front/front_left_45/front_right_45/side_left/side_right/top/unknown).
The two are not merged because their inputs (single curated photos vs. a
dense, noisy video frame sequence) and output vocabularies differ.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.video_extraction.schemas import ViewLabel


def classify_frame(frame: np.ndarray, mask: np.ndarray) -> tuple[ViewLabel, float]:
    """Classify one frame's viewing angle from its glasses mask geometry.

    Returns (label, confidence in [0, 1]). Confidence is a heuristic
    measure of how cleanly the mask matched one bucket's geometric
    signature, not a calibrated probability.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ViewLabel.UNKNOWN, 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return ViewLabel.UNKNOWN, 0.0

    aspect_ratio = float(w) / h
    mid_x = x + w // 2
    left_half = mask[y:y + h, x:mid_x]
    right_half = mask[y:y + h, mid_x:x + w]
    left_area = float(np.count_nonzero(left_half))
    right_area = float(np.count_nonzero(right_half))
    total_area = max(left_area + right_area, 1.0)
    symmetry = min(left_area, right_area) / max(max(left_area, right_area), 1.0)
    # Positive => right half has more mass (camera has swung toward showing
    # more of the wearer's/product's right side); negative => left half.
    asymmetry_signed = (right_area - left_area) / total_area

    # ---- Top view: looking down the temples, mask is short and wide-ish
    # the OTHER way (narrow vertically relative to width is actually what a
    # front view looks like too, so top is distinguished by being unusually
    # SHORT/squat: aspect ratio below ~0.95 means taller-than-wide, which a
    # front/side silhouette of glasses essentially never is).
    if aspect_ratio < 0.95:
        return ViewLabel.TOP, min(1.0, (0.95 - aspect_ratio) * 2.0 + 0.4)

    # ---- True side profile: very elongated silhouette (mostly temple arm).
    if aspect_ratio >= 4.3:
        confidence = min(1.0, (aspect_ratio - 4.3) / 3.0 + 0.5)
        return (ViewLabel.SIDE_RIGHT if asymmetry_signed >= 0 else ViewLabel.SIDE_LEFT), confidence

    # ---- Clean front view: moderate aspect ratio and good left/right symmetry.
    if 1.4 <= aspect_ratio < 3.6 and symmetry >= 0.72:
        return ViewLabel.FRONT, min(1.0, symmetry)

    # ---- 45-degree views: elongated-ish with a clear asymmetry, but not as
    # extreme as a true side profile.
    if 1.8 <= aspect_ratio < 4.3 and symmetry < 0.72:
        confidence = min(1.0, (0.72 - symmetry) * 1.5 + 0.3)
        return (ViewLabel.FRONT_RIGHT_45 if asymmetry_signed >= 0 else ViewLabel.FRONT_LEFT_45), confidence

    # Didn't match a clean bucket signature confidently.
    return ViewLabel.UNKNOWN, max(0.0, 0.5 - abs(symmetry - 0.7))
