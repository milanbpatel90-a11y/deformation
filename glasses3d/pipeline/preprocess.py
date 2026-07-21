"""Image loading, background removal, and deterministic part segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
PARTS = ("frame_front", "left_lens", "right_lens", "left_temple", "right_temple", "bridge", "nose_pads")


def _load_rgba(path: str | Path, min_size: int = 512) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if min(image.size) < min_size:
        raise ValueError(f"{path} must be at least {min_size}px on its shortest side; got {image.size}")
    return np.asarray(image)


def _remove_background(image: np.ndarray) -> np.ndarray:
    try:
        from rembg import remove
        return np.asarray(remove(Image.fromarray(image, "RGBA")))
    except ImportError:
        LOGGER.warning("rembg is not installed; using alpha/color segmentation fallback")
    except Exception as exc:
        LOGGER.warning("Background removal failed (%s); using fallback", exc)
    result = image.copy()
    if result.shape[-1] == 4 and np.all(result[..., 3] == 0):
        result[..., 3] = 255
    return result


def _foreground_mask(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3]
    if np.count_nonzero(alpha < 245) > rgba.shape[0] * rgba.shape[1] * 0.01:
        mask = alpha > 20
    else:
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        background = float(np.median(border))
        mask = np.abs(gray.astype(float) - background) > max(12.0, np.std(border) * 2.0)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask > 0).astype(np.uint8) * 255


def _parts(rgba: np.ndarray, view: str) -> dict[str, np.ndarray]:
    mask = _foreground_mask(rgba)
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        xs, ys = np.array([width // 2]), np.array([height // 2])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    parts = {name: np.zeros_like(mask) for name in PARTS}
    if view == "side":
        parts["left_temple"] = mask
        parts["right_temple"] = mask.copy()
        parts["frame_front"] = mask.copy()
        return parts
    mid = (x0 + x1) // 2
    center_band = max(4, int((x1 - x0) * 0.08))
    left = mask.copy(); left[:, mid:] = 0
    right = mask.copy(); right[:, :mid] = 0
    left_inner = left.copy(); left_inner[:, : max(x0, mid - center_band)] = 0
    right_inner = right.copy(); right_inner[:, min(x1, mid + center_band):] = 0
    parts["left_lens"], parts["right_lens"] = left_inner, right_inner
    parts["frame_front"] = mask
    bridge = np.zeros_like(mask); bridge[y0:y1 + 1, max(x0, mid - center_band):min(x1 + 1, mid + center_band)] = mask[y0:y1 + 1, max(x0, mid - center_band):min(x1 + 1, mid + center_band)]
    parts["bridge"] = bridge
    parts["nose_pads"] = bridge.copy()
    parts["left_temple"] = left.copy(); parts["right_temple"] = right.copy()
    return parts


def preprocess(front_path: str | Path, side_path: str | Path | None = None, min_size: int = 512) -> dict[str, Any]:
    front = _remove_background(_load_rgba(front_path, min_size))
    side = _remove_background(_load_rgba(side_path, min_size)) if side_path else None
    return {
        "front_rgba": front,
        "side_rgba": side,
        "front_masks": _parts(front, "front"),
        "side_masks": _parts(side, "side") if side is not None else None,
    }
