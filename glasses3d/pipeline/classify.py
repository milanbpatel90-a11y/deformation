"""Frame shape classification with a lightweight contour heuristic."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

FRAME_TYPES = ("aviator", "wayfarer", "round", "rectangle", "cat_eye", "rimless", "semi_rimless")


def classify_frame(front_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = front_masks["frame_front"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"frame_type": "rectangle", "confidence": 0.1}
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    ratio = width / max(height, 1)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * cv2.contourArea(contour) / max(perimeter * perimeter, 1)
    if circularity > 0.72:
        kind, confidence = "round", min(0.95, circularity)
    elif ratio > 2.8:
        kind, confidence = "rectangle", 0.68
    elif ratio < 1.45:
        kind, confidence = "aviator", 0.54
    else:
        kind, confidence = "wayfarer", 0.52
    return {"frame_type": kind, "confidence": float(confidence), "candidates": list(FRAME_TYPES)}
