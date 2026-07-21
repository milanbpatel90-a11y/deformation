"""PBR material descriptors sampled from image regions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _color(image: np.ndarray, mask: np.ndarray) -> list[float]:
    pixels = image[..., :3][mask > 0]
    if len(pixels) == 0: return [0.12, 0.12, 0.12, 1.0]
    return (np.median(pixels, axis=0) / 255.0).tolist() + [1.0]


def build_materials(rgba: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {"frame": {"baseColorFactor": _color(rgba, masks["frame_front"]), "metallicFactor": 0.15, "roughnessFactor": 0.32}, "lens": {"baseColorFactor": _color(rgba, masks["left_lens"]), "metallicFactor": 0.0, "roughnessFactor": 0.12, "transmissionFactor": 0.82, "ior": 1.5, "alphaMode": "BLEND"}}
