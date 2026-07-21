"""Create a compact 1024px atlas from the front image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def create_texture_atlas(rgba: np.ndarray, output_path: str | Path, size: int = 1024) -> Path:
    image = cv2.resize(rgba[..., :3], (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    Image.fromarray(image).save(output_path, format="PNG", optimize=True)
    return Path(output_path)
