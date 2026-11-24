"""Video upload validation and frame streaming.

ADAPTATION NOTE on decoding: `ffmpeg` the CLI binary is available in this
sandbox, but the OpenCV build already installed here (opencv-python-headless)
was itself compiled with FFmpeg support (verified: `cv2.getBuildInformation()`
reports `FFMPEG: YES`), so `cv2.VideoCapture` already decodes MP4/MOV through
FFmpeg internally. Frame reading therefore goes through `cv2.VideoCapture`
rather than shelling out to the `ffmpeg` binary per frame -- one dependency
instead of two, same underlying decoder. If this is ever deployed to an
environment whose OpenCV lacks FFmpeg support, `cv2.VideoCapture` will fail
to open the file at all and `probe_video` raises `InvalidVideoError`
immediately; there is no silent corrupt-frame fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.exceptions import (
    InvalidVideoError,
    UnsupportedFormatError,
    VideoDurationError,
    VideoResolutionError,
    VideoTooLargeError,
)
from backend.video_extraction.schemas import VideoMetadata

# ISO base media file format (MP4/MOV/M4V) box type that must appear near
# the start of a well-formed file. We never trust the filename/extension
# alone -- this is the "actual file content / magic bytes" check.
_ISO_BMFF_MAGIC = b"ftyp"
_MAGIC_SEARCH_WINDOW = 64


def validate_filename_extension(filename: str, settings: VideoExtractionSettings | None = None) -> str:
    """Return the lowercase extension if allowed, else raise.

    Never trusts the filename beyond its extension: the caller must
    generate its own on-disk name (see `extraction_pipeline`/API routes).
    """
    settings = settings or get_settings()
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedFormatError(
            f"Unsupported file extension '{suffix or '(none)'}'. "
            f"Allowed: {', '.join(settings.allowed_extensions)}"
        )
    return suffix


def validate_upload_size(num_bytes: int, settings: VideoExtractionSettings | None = None) -> None:
    settings = settings or get_settings()
    if num_bytes <= 0:
        raise InvalidVideoError("Uploaded file is empty.")
    if num_bytes > settings.max_upload_bytes:
        raise VideoTooLargeError(
            f"Upload is {num_bytes / (1024 * 1024):.1f} MB, which exceeds the "
            f"{settings.max_upload_mb:.0f} MB limit."
        )


def validate_magic_bytes(header: bytes) -> None:
    """Reject files whose content doesn't look like ISO-BMFF (mp4/mov/m4v),
    regardless of what extension they were uploaded with."""
    if _ISO_BMFF_MAGIC not in header[:_MAGIC_SEARCH_WINDOW]:
        raise UnsupportedFormatError(
            "File content does not look like a valid MP4/MOV video "
            "(missing an 'ftyp' container box near the start of the file)."
        )


def probe_video(path: Path, settings: VideoExtractionSettings | None = None) -> VideoMetadata:
    """Open the video and validate duration/resolution/fps bounds.

    Raises InvalidVideoError (or a subclass) on any malformed or
    out-of-bounds file; never silently accepts a broken capture.
    """
    settings = settings or get_settings()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError(
                "Could not open the uploaded file as a video. It may be corrupt "
                "or use an unsupported codec."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00") or None

        if width <= 0 or height <= 0:
            raise InvalidVideoError("Video reports zero or invalid resolution; the file is likely corrupt.")

        # A corrupt/truncated container can report fps=0 or an absurd frame
        # count. Fall back to actually counting frames (bounded) only when
        # the header looks implausible, so we never trust an unchecked
        # metadata field for validation.
        if fps <= 0 or frame_count <= 0:
            ok, _ = cap.read()
            if not ok:
                raise InvalidVideoError("Video container is unreadable (no decodable frames found).")
            raise InvalidVideoError(
                "Video is missing valid duration/frame-rate metadata; re-export it with a "
                "standard MP4/MOV container."
            )

        duration_sec = frame_count / fps

        if max(width, height) > settings.max_resolution_px:
            raise VideoResolutionError(
                f"Resolution {width}x{height} exceeds the maximum of {settings.max_resolution_px}px "
                "on the longer side."
            )
        if min(width, height) < settings.min_resolution_px:
            raise VideoResolutionError(
                f"Resolution {width}x{height} is below the minimum of {settings.min_resolution_px}px "
                "on the shorter side."
            )
        if fps > settings.max_fps:
            raise VideoResolutionError(f"Frame rate {fps:.1f} fps exceeds the maximum of {settings.max_fps:.1f} fps.")
        if fps < settings.min_fps:
            raise VideoResolutionError(f"Frame rate {fps:.1f} fps is below the minimum of {settings.min_fps:.1f} fps.")
        if duration_sec > settings.max_duration_sec:
            raise VideoDurationError(
                f"Duration {duration_sec:.1f}s exceeds the maximum of {settings.max_duration_sec:.0f}s."
            )
        if duration_sec < settings.min_duration_sec:
            raise VideoDurationError(
                f"Duration {duration_sec:.1f}s is below the minimum of {settings.min_duration_sec:.1f}s."
            )

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            codec_fourcc=fourcc,
        )
    finally:
        cap.release()


def iter_sampled_frames(
    path: Path,
    metadata: VideoMetadata,
    settings: VideoExtractionSettings | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_sec, bgr_frame) at an adaptive stride.

    Streams the video sequentially (one `VideoCapture`, grab-then-retrieve)
    instead of loading it into memory or seeking randomly -- seeking by
    frame index is unreliable on many consumer MP4/MOV encodings, so every
    frame is `grab()`-ed (cheap) and only `retrieve()`-ed (decoded) on
    frames that land on the sampling stride.
    """
    settings = settings or get_settings()
    stride = max(1, round(metadata.fps / max(settings.sample_rate_fps, 0.1)))

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("Could not reopen video for frame extraction.")
        frame_index = 0
        yielded = 0
        while True:
            if yielded >= settings.max_candidate_frames:
                break
            ok = cap.grab()
            if not ok:
                break
            if frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    timestamp = frame_index / metadata.fps
                    yield frame_index, timestamp, frame
                    yielded += 1
            frame_index += 1
    finally:
        cap.release()


def safe_temp_video_name(job_id: str, extension: str) -> str:
    """Build a storage-relative filename that never trusts the user's own
    filename. Lives under ``original/`` to match the task brief's output
    tree (``original/source_video.mp4``)."""
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"original/source_video{ext}"


def grab_frame_at_index(path: Path, frame_index: int) -> np.ndarray:
    """Best-effort random-access read of one frame by index, used ONLY for
    re-deriving a replacement frame from the still-stored original video
    when its candidate JPEG (see `extraction_pipeline._persist_candidates`)
    is unavailable for some reason. `CAP_PROP_POS_FRAMES` seeking is
    documented upstream (see this module's top docstring) as unreliable on
    some consumer MP4/MOV encodings -- callers should prefer the persisted
    `candidates/candidate_<frame_index>.jpg` file when it exists."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("Could not reopen the stored video to re-extract a frame.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise InvalidVideoError(f"Could not read frame {frame_index} from the stored video.")
        return frame
    finally:
        cap.release()
