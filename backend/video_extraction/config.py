"""Configuration for the video-to-reference extraction feature.

Every tunable lives here as an environment variable with a documented,
sane default -- nothing in the rest of this package hardcodes a magic
number. See backend/video_extraction/README.md for the full reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class VideoExtractionSettings:
    """All configurable limits/thresholds for the video extraction pipeline.

    Every field has an environment-variable override (see the class body)
    and a documented default: nothing here is a silent magic number.
    """

    # ---- Storage / persistence ---------------------------------------
    storage_root: Path = field(default_factory=lambda: _resolve_path(
        _env_str("VIDEOX_STORAGE_ROOT", "storage/video_extraction")
    ))
    db_path: Path = field(default_factory=lambda: _resolve_path(
        _env_str("VIDEOX_DB_PATH", "data/video_extraction.sqlite3")
    ))

    # ---- Upload validation ----------------------------------------------
    max_upload_mb: float = field(default_factory=lambda: _env_float("VIDEOX_MAX_UPLOAD_MB", 300.0))
    max_duration_sec: float = field(default_factory=lambda: _env_float("VIDEOX_MAX_DURATION_SEC", 120.0))
    min_duration_sec: float = field(default_factory=lambda: _env_float("VIDEOX_MIN_DURATION_SEC", 1.5))
    max_resolution_px: int = field(default_factory=lambda: _env_int("VIDEOX_MAX_RESOLUTION", 4096))
    min_resolution_px: int = field(default_factory=lambda: _env_int("VIDEOX_MIN_RESOLUTION", 180))
    max_fps: float = field(default_factory=lambda: _env_float("VIDEOX_MAX_FPS", 120.0))
    min_fps: float = field(default_factory=lambda: _env_float("VIDEOX_MIN_FPS", 4.0))
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v")

    # ---- Adaptive frame sampling ----------------------------------------
    sample_rate_fps: float = field(default_factory=lambda: _env_float("VIDEOX_SAMPLE_RATE_FPS", 3.0))
    max_candidate_frames: int = field(default_factory=lambda: _env_int("VIDEOX_MAX_CANDIDATE_FRAMES", 180))
    motion_change_threshold: float = field(
        default_factory=lambda: _env_float("VIDEOX_MOTION_CHANGE_THRESHOLD", 6.0)
    )

    # ---- Quality analysis -------------------------------------------------
    sharpness_reference_width: int = field(
        default_factory=lambda: _env_int("VIDEOX_SHARPNESS_REFERENCE_WIDTH", 800)
    )
    sharpness_min: float = field(default_factory=lambda: _env_float("VIDEOX_SHARPNESS_MIN", 45.0))
    exposure_dark_fraction_max: float = field(
        default_factory=lambda: _env_float("VIDEOX_EXPOSURE_DARK_FRACTION_MAX", 0.6)
    )
    exposure_bright_fraction_max: float = field(
        default_factory=lambda: _env_float("VIDEOX_EXPOSURE_BRIGHT_FRACTION_MAX", 0.6)
    )
    glare_fraction_max: float = field(default_factory=lambda: _env_float("VIDEOX_GLARE_FRACTION_MAX", 0.35))

    # ---- Detection filtering -----------------------------------------------
    min_glasses_area_fraction: float = field(
        default_factory=lambda: _env_float("VIDEOX_MIN_GLASSES_AREA_FRACTION", 0.015)
    )
    max_glasses_area_fraction: float = field(
        default_factory=lambda: _env_float("VIDEOX_MAX_GLASSES_AREA_FRACTION", 0.95)
    )

    # ---- Duplicate filtering / ranking --------------------------------------
    duplicate_hash_threshold: int = field(default_factory=lambda: _env_int("VIDEOX_DUPLICATE_HASH_THRESHOLD", 6))
    # Superseded by the per-view target-count table in schemas.TARGET_VIEW_COUNTS
    # (see frame_ranker.py); kept only so an old override doesn't hard-fail.
    max_selected_per_view: int = field(default_factory=lambda: _env_int("VIDEOX_MAX_SELECTED_PER_VIEW", 2))
    max_total_selected: int = field(default_factory=lambda: _env_int("VIDEOX_MAX_TOTAL_SELECTED", 12))

    # ---- Background job execution --------------------------------------------
    worker_pool_size: int = field(default_factory=lambda: _env_int("VIDEOX_WORKER_POOL_SIZE", 2))
    max_jobs_per_user: int = field(default_factory=lambda: _env_int("VIDEOX_MAX_JOBS_PER_USER", 3))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


_settings: VideoExtractionSettings | None = None


def get_settings() -> VideoExtractionSettings:
    """Return the process-wide settings instance (cheap to call; cached)."""
    global _settings
    if _settings is None:
        _settings = VideoExtractionSettings()
    return _settings


def reset_settings_cache() -> None:
    """Test hook: force settings to be re-read from the environment."""
    global _settings
    _settings = None
