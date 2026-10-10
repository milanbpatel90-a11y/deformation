"""Reusable orbit-video fixtures for tests.

An orbit clip is synthesised from a real eyewear photo rather than drawn
shapes, because the pipeline runs a real YOLO model: a synthetic line drawing
would never be segmented, so an end-to-end test built from one would prove
nothing. Horizontally foreshortening a real photo imitates the frame turning
away from square-on, and the trained model detects the result at every scale
used here.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = PROJECT_ROOT / "test_images"
CANVAS_HEIGHT, CANVAS_WIDTH = 720, 1280
BACKDROP = (235, 235, 235)

#: A photo the one-class eyewear model reliably detects.
DEFAULT_PHOTO = "test_004_metal.jpg"
#: Scales at the extremes of the sweep: square-on and near-profile.
SQUARE_ON_SCALE = 1.0
PROFILE_SCALE = 0.35


def test_image_path(name: str = DEFAULT_PHOTO) -> Path:
    return TEST_IMAGES / name


def load_photo(name: str = DEFAULT_PHOTO) -> np.ndarray:
    image = cv2.imread(str(test_image_path(name)))
    if image is None:
        raise FileNotFoundError(f"Missing fixture image: {test_image_path(name)}")
    return image


def orbit_frame(
    photo: np.ndarray,
    scale: float,
    angle: float = 0.0,
    canvas: tuple[int, int] = (CANVAS_HEIGHT, CANVAS_WIDTH),
    backdrop: tuple[int, int, int] = BACKDROP,
) -> np.ndarray:
    """Foreshorten and tilt a product photo onto a plain backdrop."""
    height, width = photo.shape[:2]
    frame = np.full((canvas[0], canvas[1], 3), backdrop, dtype=np.uint8)
    target_width = max(8, int(width * scale))
    resized = cv2.resize(photo, (target_width, height), interpolation=cv2.INTER_AREA)
    if angle:
        matrix = cv2.getRotationMatrix2D((target_width / 2.0, height / 2.0), angle, 1.0)
        resized = cv2.warpAffine(
            resized, matrix, (target_width, height), borderValue=backdrop
        )
    y0 = (canvas[0] - height) // 2
    x0 = (canvas[1] - target_width) // 2
    frame[y0 : y0 + height, x0 : x0 + target_width] = resized
    return frame


def orbit_frames(
    count: int,
    photo_name: str = DEFAULT_PHOTO,
    square_on: float = SQUARE_ON_SCALE,
    profile: float = PROFILE_SCALE,
    tilt: float = 10.0,
) -> list[np.ndarray]:
    """A full 360-degree sweep: square-on, through profile, back to square-on."""
    photo = load_photo(photo_name)
    frames: list[np.ndarray] = []
    for index in range(count):
        theta = 2.0 * math.pi * index / max(count, 1)
        scale = profile + (square_on - profile) * (0.5 * (1.0 + math.cos(theta)))
        angle = tilt * math.sin(theta)
        frames.append(orbit_frame(photo, scale, angle))
    return frames


def blur_frame(frame: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    kernel = int(max(3, 2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(frame, (kernel, kernel), sigma)


def glare_frame(frame: np.ndarray, fraction: float = 0.12) -> np.ndarray:
    """Blow out a square patch, imitating a specular reflection."""
    out = frame.copy()
    height, width = out.shape[:2]
    side = int(math.sqrt(fraction * height * width))
    x0 = width // 2 - side // 2
    y0 = height // 2 - side // 2
    out[y0 : y0 + side, x0 : x0 + side] = 255
    return out


def darken_frame(frame: np.ndarray, scale: float = 0.2) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def scratch_dir(name: str = "_orbit_video_tests") -> Path:
    """A project-local scratch directory for generated clips.

    OpenCV's path-based file APIs fail on Windows 8.3 short paths such as
    ``C:\\Users\\PETPOO~1\\AppData\\Local\\Temp``, which is what the system temp
    directory resolves to on this machine. Test artefacts are therefore written
    under the project, exactly as the API writes its uploads.
    """
    path = PROJECT_ROOT / "output" / "development" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_video(path: Path, frames: list[np.ndarray], fps: float = 20.0) -> Path:
    """Write frames as an mp4v MP4, the codec OpenCV ships portably."""
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open an mp4v VideoWriter at {path}. This usually means the "
            "path is a Windows 8.3 short path (for example PETPOO~1); write to a "
            "project-local directory such as scratch_dir() instead."
        )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def orbit_video(
    path: Path,
    seconds: float = 10.0,
    fps: float = 20.0,
    photo_name: str = DEFAULT_PHOTO,
) -> Path:
    """Render a complete synthetic orbit clip of the requested duration."""
    count = max(4, int(round(seconds * fps)))
    return write_video(path, orbit_frames(count, photo_name), fps)
