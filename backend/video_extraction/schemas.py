"""Validated configuration and stable view names."""
from __future__ import annotations
import os
from dataclasses import dataclass

VIEW_NAMES = ("front", "front_left_45", "front_right_45", "side_left", "side_right", "top", "unknown")

@dataclass(frozen=True)
class VideoLimits:
    max_upload_bytes: int = 150 * 1024 * 1024
    max_duration_seconds: float = 90.0
    min_width: int = 480
    min_height: int = 360
    max_width: int = 3840
    max_height: int = 2160
    max_fps: float = 60.0
    sample_interval_seconds: float = 0.35
    max_sample_frames: int = 240
    max_selected_frames: int = 24
    min_detection_confidence: float = 0.35
    min_quality_score: float = 0.34
    min_bbox_coverage: float = 0.012
    max_bbox_coverage: float = 0.92
    duplicate_hash_distance: int = 6
    def __post_init__(self):
        if self.max_upload_bytes <= 0 or self.max_duration_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("Video limits must be positive.")
    @classmethod
    def from_env(cls):
        return cls(
            max_upload_bytes=int(os.getenv("VIDEO_MAX_UPLOAD_BYTES", str(150*1024*1024))),
            max_duration_seconds=float(os.getenv("VIDEO_MAX_DURATION_SECONDS", "90")),
            sample_interval_seconds=float(os.getenv("VIDEO_SAMPLE_INTERVAL_SECONDS", "0.35")),
            max_sample_frames=int(os.getenv("VIDEO_MAX_SAMPLE_FRAMES", "240")),
            max_selected_frames=int(os.getenv("VIDEO_MAX_SELECTED_FRAMES", "24")),
            min_detection_confidence=float(os.getenv("VIDEO_MIN_DETECTION_CONFIDENCE", "0.35")),
        )
