"""Normalized landmark extraction from masks, with optional model hooks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.25, 0.25, 0.75, 0.75)
    h, w = mask.shape
    return (float(xs.min() / w), float(ys.min() / h), float(xs.max() / w), float(ys.max() / h))


def detect_landmarks(front_masks: dict[str, np.ndarray], side_masks: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    left = _box(front_masks["left_lens"]); right = _box(front_masks["right_lens"]); frame = _box(front_masks["frame_front"])
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    mid_x = (lx + rx) / 2
    result: dict[str, Any] = {"front": {
        "lens_center_L": [lx, ly], "lens_center_R": [rx, ry],
        "lens_outer_L": [left[0], ly], "lens_inner_L": [left[2], ly],
        "lens_inner_R": [right[0], ry], "lens_outer_R": [right[2], ry],
        "bridge_midpoint": [mid_x, (ly + ry) / 2],
        "hinge_L": [left[0], (left[1] + left[3]) / 2], "hinge_R": [right[2], (right[1] + right[3]) / 2],
        "frame_top": [mid_x, frame[1]], "frame_bottom": [mid_x, frame[3]],
    }}
    if side_masks:
        side = _box(side_masks["left_temple"])
        result["side"] = {"temple_start": [side[0], (side[1] + side[3]) / 2], "temple_end": [side[2], (side[1] + side[3]) / 2], "temple_bend": [(side[0] + side[2]) / 2, side[3]], "pantoscopic_tilt_deg": 8.0}
    else:
        result["side"] = None
    return result
