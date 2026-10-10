"""Decode a 360-degree orbit video into a bounded, evenly sampled frame set.

Decoding is bounded in three independent ways so that a hostile or simply huge
upload cannot exhaust memory or CPU:

* ``max_frames`` caps how many sampled frames are ever held in memory,
* ``long_edge`` caps the resolution of each retained frame,
* the recommended/hard duration windows reject clips that cannot contain a
  usable orbit before a single frame is kept.

Only frames on the sample grid are ``retrieve``\\ d; the rest are skipped with
``grab``, which is what makes sampling a 60 fps clip cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

#: Capture window that gives a complete orbit without a needlessly long clip.
RECOMMENDED_MIN_SECONDS = 8.0
RECOMMENDED_MAX_SECONDS = 12.0
#: Outside these bounds the clip is rejected outright.
HARD_MIN_SECONDS = 2.0
HARD_MAX_SECONDS = 90.0

#: Sample grid. 6 fps over 8-12 s yields 48-72 candidates, comfortably more
#: than the 18-24 views the selector keeps.
DEFAULT_TARGET_FPS = 6.0
MAX_TARGET_FPS = 30.0
MAX_DECODED_FRAMES = 240
MAX_LONG_EDGE = 1280
#: Fewer samples than this cannot describe an orbit at all, so the decode is
#: treated as a failure rather than passed downstream. This is also what catches
#: a file that is not really a video: FFmpeg will decode a bare JPEG as a
#: one-frame MJPEG stream, so "the container opened" is not proof of a clip.
MIN_USABLE_FRAMES = 6
#: Sampling fewer than this is workable but leaves little for the selector.
RECOMMENDED_DECODED_FRAMES = 12

#: Containers accepted for upload. OpenCV decodes these through its bundled
#: FFmpeg build; the list is an allow-list, not a capability probe.
ALLOWED_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"})


@dataclass(slots=True)
class VideoInfo:
    """Container-level facts about a decoded clip."""

    path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    duration_known: bool

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "duration_known": self.duration_known,
        }


@dataclass(slots=True)
class Frame:
    """One retained sample from the clip."""

    index: int          # position in the sampled sequence (stable id downstream)
    source_index: int   # frame number in the source clip
    timestamp: float    # seconds into the clip
    image: np.ndarray   # BGR uint8


@dataclass(slots=True)
class ExtractionResult:
    frames: list[Frame]
    info: VideoInfo
    sampled_fps: float
    source_step: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": self.info.to_dict(),
            "decoded_frames": len(self.frames),
            "sampled_fps": round(self.sampled_fps, 3),
            "source_step": self.source_step,
            "warnings": list(self.warnings),
        }


class FrameExtractor:
    """Turn an orbit video into a bounded list of :class:`Frame` samples."""

    def __init__(
        self,
        target_fps: float = DEFAULT_TARGET_FPS,
        max_frames: int = MAX_DECODED_FRAMES,
        long_edge: int = MAX_LONG_EDGE,
    ) -> None:
        self.target_fps = float(min(max(target_fps, 0.1), MAX_TARGET_FPS))
        self.max_frames = int(max(1, min(max_frames, MAX_DECODED_FRAMES)))
        self.long_edge = int(max(64, min(long_edge, MAX_LONG_EDGE)))

    # ── container inspection ────────────────────────────────────────────────
    @staticmethod
    def probe(path: Path | str) -> VideoInfo:
        """Read container properties without decoding frames."""
        path = Path(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(
                "Could not open the video. Re-export it as an H.264 MP4 and try again."
            )
        try:
            return FrameExtractor._read_info(capture, path)
        finally:
            capture.release()

    @staticmethod
    def _read_info(capture: cv2.VideoCapture, path: Path) -> VideoInfo:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some containers report 0/nan fps; 30 is the least surprising default
        # and only affects timestamp bookkeeping, never the sample stride when
        # the reported rate is unusable.
        duration_known = True
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 30.0
            duration_known = False
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            duration_known = False
        duration = frame_count / fps if duration_known else 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return VideoInfo(
            path=str(path),
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            width=width,
            height=height,
            duration_known=duration_known,
        )

    @staticmethod
    def _duration_warnings(info: VideoInfo) -> list[str]:
        """Reject unusable clips, and warn about merely unhelpful ones."""
        warnings: list[str] = []
        if info.duration_known:
            if info.duration_seconds < HARD_MIN_SECONDS:
                raise ValueError(
                    f"Orbit video is {info.duration_seconds:.1f}s long; "
                    f"record at least {HARD_MIN_SECONDS:.0f}s (8-12s recommended)."
                )
            if info.duration_seconds > HARD_MAX_SECONDS:
                raise ValueError(
                    f"Orbit video is {info.duration_seconds:.0f}s long; "
                    f"trim it to {RECOMMENDED_MIN_SECONDS:.0f}-{RECOMMENDED_MAX_SECONDS:.0f}s."
                )
            if info.duration_seconds < RECOMMENDED_MIN_SECONDS:
                warnings.append(
                    f"Clip is {info.duration_seconds:.1f}s; "
                    f"{RECOMMENDED_MIN_SECONDS:.0f}-{RECOMMENDED_MAX_SECONDS:.0f}s gives fuller coverage."
                )
            elif info.duration_seconds > RECOMMENDED_MAX_SECONDS:
                warnings.append(
                    f"Clip is {info.duration_seconds:.1f}s; "
                    f"{RECOMMENDED_MIN_SECONDS:.0f}-{RECOMMENDED_MAX_SECONDS:.0f}s is enough."
                )
        else:
            warnings.append("Container did not report a reliable duration; duration limits were not enforced.")
        if info.width and info.height and max(info.width, info.height) < 480:
            warnings.append(f"Video is only {info.width}x{info.height}; frame detail will limit measurement accuracy.")
        return warnings

    # ── decoding ────────────────────────────────────────────────────────────
    def extract(self, path: Path | str) -> ExtractionResult:
        """Decode ``path`` into evenly spaced, downscaled samples."""
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Video not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(
                "Could not decode the video. Re-export it as an H.264 MP4 and try again."
            )
        try:
            info = self._read_info(capture, path)
            warnings = self._duration_warnings(info)
            step = max(1, int(round(info.fps / self.target_fps)))

            frames: list[Frame] = []
            source_index = 0
            while len(frames) < self.max_frames:
                if not capture.grab():
                    break
                if source_index % step == 0:
                    ok, image = capture.retrieve()
                    if ok and image is not None:
                        frames.append(
                            Frame(
                                index=len(frames),
                                source_index=source_index,
                                timestamp=source_index / info.fps if info.fps > 0 else 0.0,
                                image=self._downscale(image),
                            )
                        )
                source_index += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError("No frames could be decoded from the video.")

        if len(frames) < MIN_USABLE_FRAMES:
            raise ValueError(
                f"Only {len(frames)} frame(s) could be decoded from the upload. The file does not "
                "appear to be a video, or its codec is unsupported. Re-export it as an H.264 MP4."
            )
        if len(frames) < RECOMMENDED_DECODED_FRAMES:
            warnings.append(
                f"Only {len(frames)} frames were sampled; "
                f"at least {RECOMMENDED_DECODED_FRAMES} are needed to select 18-24 views. "
                "Record a slightly longer orbit."
            )
        if len(frames) >= self.max_frames:
            warnings.append(f"Frame decoding stopped at the {self.max_frames}-frame cap.")

        sampled_fps = info.fps / step
        LOGGER.info(
            "Orbit decode: %s frame(s) sampled at %.2f fps from %.2fs/%s fps source",
            len(frames), sampled_fps, info.duration_seconds, round(info.fps, 2),
        )
        return ExtractionResult(
            frames=frames,
            info=info,
            sampled_fps=sampled_fps,
            source_step=step,
            warnings=warnings,
        )

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.long_edge:
            return image
        scale = self.long_edge / float(longest)
        return cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
