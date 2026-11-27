"""Heuristic viewing-angle classification for video frames.

IMPORTANT, stated plainly (see also the package README): this is a
heuristic classifier built from mask geometry (bounding-box aspect ratio,
left/right mask-area symmetry) layered on top of the YOLO segmentation.
It is NOT a calibrated pose estimator and does not know the true camera
rotation in degrees. "front_left_45" / "front_right_45" / "left_side" /
"right_side" / "top" / "bottom" / "rear_inside" are approximate buckets chosen from aspect-ratio and
asymmetry thresholds, not a geometric reconstruction of viewing angle.
When confidence is low the frame is labeled `unknown` rather than guessed,
and the dashboard lets a human confirm/correct it before approval.

This is distinct from `backend.fusion.view_classifier.ViewClassifier`,
which classifies single uploaded product photos into a coarser
front/side/top/perspective set for the existing multi-image pipeline.
This module classifies frames sampled from a *video*, over time, into the
finer-grained bucket set the video-extraction brief asks for
(front/front_left_45/front_right_45/left_side/right_side/top/unknown, plus
`bottom` from a narrow, documented component-evidence rule below).
The two are not merged because their inputs (single curated photos vs. a
dense, noisy video frame sequence) and output vocabularies differ.

`bottom` and `rear_inside`: the brief's full view table also asks for these.
Mask aspect ratio alone cannot tell "looking from below" apart from "looking
from above" (`top`), and nothing in a bounding box distinguishes "looking at
the outside of the temples" from "looking at the inside" (`rear_inside`).
Rather than guess, this module only ever emits `bottom` when per-class
segmentation evidence specifically supports it (see `classify_frame`'s
`per_class_masks` argument): a squat silhouette where the nose-pad
component's area is unusually large relative to the rim is a reasonable,
documented signal that the camera is looking up at the underside, because
nose pads are typically only prominent from that angle. No such signal
exists yet for `rear_inside` in this model's 6 classes, so that bucket is
reachable only through the merchant's manual view-correction endpoint
(`POST /{job_id}/frames/{frame_index}/correct-view`) -- this is a stated
limitation, not an oversight (see the package README).
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.video_extraction.schemas import ViewLabel

# A nose-pad-to-rim area ratio above this, combined with a squat (top-like)
# silhouette, is treated as weak-but-real evidence of a bottom/underside
# view rather than a top view. Conservative by design: this only fires when
# segmentation actually found both components, never from bbox shape alone.
_BOTTOM_NOSE_PAD_RATIO = 0.35


def classify_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    per_class_masks: dict[str, np.ndarray] | None = None,
) -> tuple[ViewLabel, float]:
    """Classify one frame's viewing angle from its glasses mask geometry.

    Returns (label, confidence in [0, 1]). Confidence is a heuristic
    measure of how cleanly the mask matched one bucket's geometric
    signature, not a calibrated probability. `per_class_masks` (optional)
    supplies the component-level evidence used for the narrow `bottom` rule
    below; pass `None` or `{}` when unavailable (e.g. the OpenCV fallback
    segmenter) and this function behaves exactly as it did before.
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

    # ---- Top/bottom: looking down (or up) the temples, mask is unusually
    # SHORT/squat (aspect ratio below ~0.95 means taller-than-wide, which a
    # front/side silhouette of glasses essentially never is). Disambiguate
    # bottom from top only via the documented nose-pad evidence rule above;
    # default to `top` (the brief's more common/expected squat-view case).
    if aspect_ratio < 0.95:
        base_confidence = min(1.0, (0.95 - aspect_ratio) * 2.0 + 0.4)
        nose_pad_mask = (per_class_masks or {}).get("nose_pad")
        rim_mask = (per_class_masks or {}).get("eyewear_rim")
        if nose_pad_mask is not None and rim_mask is not None:
            rim_area = float(np.count_nonzero(rim_mask))
            nose_pad_area = float(np.count_nonzero(nose_pad_mask))
            if rim_area > 0 and (nose_pad_area / rim_area) >= _BOTTOM_NOSE_PAD_RATIO:
                return ViewLabel.BOTTOM, min(1.0, base_confidence * 0.85)
        return ViewLabel.TOP, base_confidence

    # ---- True side profile: very elongated silhouette (mostly temple arm).
    if aspect_ratio >= 4.3:
        confidence = min(1.0, (aspect_ratio - 4.3) / 3.0 + 0.5)
        return (ViewLabel.RIGHT_SIDE if asymmetry_signed >= 0 else ViewLabel.LEFT_SIDE), confidence

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
