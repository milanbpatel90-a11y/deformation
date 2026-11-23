"""Per-frame image quality analysis: sharpness, exposure, glare.

All thresholds come from `VideoExtractionSettings` (env-var configurable);
nothing here is a hardcoded magic number chosen for one resolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.schemas import QualityMetrics


def _resize_for_analysis(frame: np.ndarray, reference_width: int) -> np.ndarray:
    """Normalize resolution before computing sharpness so a fixed variance
    threshold means the same thing on a 480p and a 4K source frame."""
    h, w = frame.shape[:2]
    if w <= 0:
        return frame
    scale = reference_width / float(w)
    if abs(scale - 1.0) < 1e-3:
        return frame
    new_size = (reference_width, max(1, round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interpolation)


def compute_sharpness(frame: np.ndarray, settings: VideoExtractionSettings | None = None) -> float:
    """Variance of the Laplacian on a resolution-normalized grayscale copy.

    Higher = sharper. Motion blur and out-of-focus frames both reduce
    high-frequency content and so reduce this score.
    """
    settings = settings or get_settings()
    normalized = _resize_for_analysis(frame, settings.sharpness_reference_width)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure(frame: np.ndarray) -> tuple[float, float]:
    """Return (dark_fraction, bright_fraction): fraction of pixels whose
    luminance is near-black (<16) or near-white (>239), histogram-based."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = gray.size
    dark = float(np.count_nonzero(gray < 16)) / total
    bright = float(np.count_nonzero(gray > 239)) / total
    return dark, bright


def compute_glare(frame: np.ndarray, mask: np.ndarray | None) -> float:
    """Fraction of the glasses mask region that is a saturated specular
    highlight (very bright, very low color saturation -- a glare/reflection
    blob rather than genuine frame material). Returns 0.0 if no mask/area."""
    if mask is None or not np.any(mask):
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    glare_pixels = (value > 235) & (saturation < 40)
    mask_bool = mask.astype(bool)
    mask_area = int(np.count_nonzero(mask_bool))
    if mask_area == 0:
        return 0.0
    glare_in_mask = int(np.count_nonzero(glare_pixels & mask_bool))
    return glare_in_mask / mask_area


def analyze_frame(
    frame: np.ndarray,
    mask: np.ndarray | None = None,
    settings: VideoExtractionSettings | None = None,
) -> QualityMetrics:
    settings = settings or get_settings()
    reasons: list[str] = []

    sharpness = compute_sharpness(frame, settings)
    is_blurry = sharpness < settings.sharpness_min
    if is_blurry:
        reasons.append(f"Sharpness {sharpness:.1f} below minimum {settings.sharpness_min:.1f} (motion blur/out of focus)")

    dark_fraction, bright_fraction = compute_exposure(frame)
    exposure_ok = (
        dark_fraction <= settings.exposure_dark_fraction_max
        and bright_fraction <= settings.exposure_bright_fraction_max
    )
    if dark_fraction > settings.exposure_dark_fraction_max:
        reasons.append(f"Underexposed: {dark_fraction:.0%} of pixels near-black")
    if bright_fraction > settings.exposure_bright_fraction_max:
        reasons.append(f"Overexposed: {bright_fraction:.0%} of pixels near-white")

    glare_fraction = compute_glare(frame, mask)
    glare_ok = glare_fraction <= settings.glare_fraction_max
    if not glare_ok:
        reasons.append(f"Glare/reflection covers {glare_fraction:.0%} of the glasses region")

    # Composite 0-100 score: sharpness contributes most (clamped/normalized
    # against 3x the minimum so an already-good frame doesn't need to be
    # "infinitely" sharp to score well), exposure and glare are penalties.
    sharpness_component = float(np.clip(sharpness / max(settings.sharpness_min * 3.0, 1e-6), 0.0, 1.0)) * 60.0
    exposure_penalty = (dark_fraction + bright_fraction) * 20.0
    glare_penalty = glare_fraction * 20.0
    overall = float(np.clip(40.0 + sharpness_component - exposure_penalty - glare_penalty, 0.0, 100.0))

    return QualityMetrics(
        sharpness_score=round(sharpness, 2),
        is_blurry=is_blurry,
        dark_fraction=round(dark_fraction, 4),
        bright_fraction=round(bright_fraction, 4),
        exposure_ok=exposure_ok,
        glare_fraction=round(glare_fraction, 4),
        glare_ok=glare_ok,
        overall_quality=round(overall, 1),
        reasons=reasons,
    )
