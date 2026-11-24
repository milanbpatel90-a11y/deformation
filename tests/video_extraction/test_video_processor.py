from __future__ import annotations

import numpy as np
import pytest

from backend.video_extraction.config import get_settings
from backend.video_extraction.exceptions import (
    InvalidVideoError,
    UnsupportedFormatError,
    VideoDurationError,
    VideoResolutionError,
    VideoTooLargeError,
)
from backend.video_extraction.video_processor import (
    probe_video,
    safe_temp_video_name,
    validate_filename_extension,
    validate_magic_bytes,
    validate_upload_size,
)


def test_valid_extensions_accepted():
    assert validate_filename_extension("glasses.mp4") == ".mp4"
    assert validate_filename_extension("glasses.MOV") == ".mov"


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFormatError):
        validate_filename_extension("glasses.avi")


def test_unsupported_extension_rejected_even_with_no_name():
    with pytest.raises(UnsupportedFormatError):
        validate_filename_extension("")


def test_upload_size_rejects_empty():
    with pytest.raises(InvalidVideoError):
        validate_upload_size(0)


def test_upload_size_rejects_oversized():
    settings = get_settings()
    with pytest.raises(VideoTooLargeError):
        validate_upload_size(settings.max_upload_bytes + 1)


def test_upload_size_accepts_within_bounds():
    validate_upload_size(1024)  # should not raise


def test_magic_bytes_rejects_non_video_content():
    with pytest.raises(UnsupportedFormatError):
        validate_magic_bytes(b"not a real video file at all, just text" * 4)


def test_magic_bytes_accepts_iso_bmff_header():
    header = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40
    validate_magic_bytes(header)  # should not raise


def test_safe_temp_video_name_ignores_user_filename():
    # Regardless of what a malicious filename says, the generated on-disk
    # name is always derived from a validated extension only, under the
    # task brief's own "original/" subfolder -- never anything influenced
    # by the user-supplied filename, and never a path that can escape it.
    name = safe_temp_video_name("abc123", ".mp4")
    assert name == "original/source_video.mp4"
    assert ".." not in name
    assert name.count("/") == 1 and name.startswith("original/")


def test_probe_video_rejects_corrupt_file(tmp_path):
    bogus = tmp_path / "corrupt.mp4"
    bogus.write_bytes(b"this is not a real video container")
    with pytest.raises(InvalidVideoError):
        probe_video(bogus)


def _write_real_test_video(path, width=320, height=240, fps=10.0, seconds=2.0, colored=True):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    n_frames = int(fps * seconds)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if colored:
            frame[:, :, 1] = (i * 5) % 255
        writer.write(frame)
    writer.release()


def test_probe_video_accepts_a_real_small_video(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_real_test_video(video_path)
    metadata = probe_video(video_path)
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.frame_count > 0
    assert metadata.duration_sec > 0


def test_probe_video_rejects_too_short_duration(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("VIDEOX_MIN_DURATION_SEC", "10")
    from backend.video_extraction import config

    config.reset_settings_cache()
    video_path = tmp_path / "short.mp4"
    _write_real_test_video(video_path, seconds=0.5)
    with pytest.raises(VideoDurationError):
        probe_video(video_path)
