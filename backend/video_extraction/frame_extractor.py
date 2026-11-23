"""Adaptive frame sampling: motion-aware, not fixed every-Nth-frame.

Reads the video once, sequentially (never the whole thing in memory --
see `video_processor.iter_sampled_frames`), sampling at the configured
target rate, and keeps a sampled frame only if it differs meaningfully
from the previously *kept* sampled frame (mean absolute pixel difference
on a small grayscale thumbnail). This avoids spending detection/quality
compute, and avoids flooding frame_ranker's buckets, on long near-static
runs (e.g. the person adjusting their grip) while still catching genuine
orientation changes between those runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.schemas import VideoMetadata
from backend.video_extraction.video_processor import iter_sampled_frames

_MOTION_THUMBNAIL_SIZE = (64, 64)


@dataclass
class RawFrame:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _MOTION_THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def extract_candidate_frames(
    path: Path,
    metadata: VideoMetadata,
    settings: VideoExtractionSettings | None = None,
) -> list[RawFrame]:
    """Return the adaptively-sampled candidate frames for a video.

    The first sampled frame is always kept (there is nothing to compare
    it to). After that, a frame is kept only when it represents a
    meaningful change from the last kept frame, OR when we haven't kept a
    frame in a while (so a slow, steady rotation isn't starved entirely
    for lack of a single dramatic jump). The "haven't kept one in a while"
    fallback fires after 2x the configured sample interval has passed
    without a keep.
    """
    settings = settings or get_settings()
    kept: list[RawFrame] = []
    last_thumbnail: np.ndarray | None = None
    last_kept_timestamp = 0.0
    min_gap_sec = 1.0 / max(settings.sample_rate_fps, 0.1)

    for frame_index, timestamp_sec, frame in iter_sampled_frames(path, metadata, settings):
        thumbnail = _thumbnail(frame)
        if last_thumbnail is None:
            keep = True
        else:
            diff = float(np.mean(np.abs(thumbnail - last_thumbnail)))
            stale = (timestamp_sec - last_kept_timestamp) > (2.0 * min_gap_sec)
            keep = diff >= settings.motion_change_threshold or stale

        if keep:
            kept.append(RawFrame(frame_index=frame_index, timestamp_sec=timestamp_sec, image=frame))
            last_thumbnail = thumbnail
            last_kept_timestamp = timestamp_sec

        if len(kept) >= settings.max_candidate_frames:
            break

    return kept
