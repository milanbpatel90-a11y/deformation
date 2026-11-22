"""Streaming video metadata inspection with ffprobe and OpenCV decode validation."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
import cv2
from .exceptions import VideoValidationError
from .schemas import VideoLimits

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
def inspect_video(path: str | Path, supplied_name: str, limits: VideoLimits) -> dict:
    file_path = Path(path)
    suffix = Path(supplied_name or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise VideoValidationError("Only MP4 and MOV video uploads are supported.")
    if not file_path.is_file() or file_path.stat().st_size < 128:
        raise VideoValidationError("Uploaded video is empty or invalid.")
    if file_path.stat().st_size > limits.max_upload_bytes:
        raise VideoValidationError("Video exceeds the configured upload size limit.")
    probe = shutil.which("ffprobe")
    if probe:
        command = [probe, "-v", "error", "-show_entries", "format=duration,format_name:stream=codec_type,codec_name,width,height,avg_frame_rate", "-of", "json", str(file_path)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            info = json.loads(completed.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise VideoValidationError("Video container or codec could not be validated.") from exc
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not streams:
            raise VideoValidationError("The uploaded file contains no video stream.")
        stream = streams[0]
        duration = float(info.get("format", {}).get("duration") or 0)
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        rate = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(rate[0]) / max(float(rate[1]), 1.0)
        codec = stream.get("codec_name", "unknown")
    else:
        capture = cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            raise VideoValidationError("OpenCV cannot decode this video. Install FFmpeg for full MP4/MOV validation.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = count / fps if fps > 0 else 0
        ok, frame = capture.read(); capture.release()
        if not ok or frame is None:
            raise VideoValidationError("Video has no decodable frames.")
        codec = "opencv-validated"
    if duration <= 0 or duration > limits.max_duration_seconds:
        raise VideoValidationError(f"Video duration must be between 0 and {limits.max_duration_seconds:g} seconds.")
    if width < limits.min_width or height < limits.min_height or width > limits.max_width or height > limits.max_height:
        raise VideoValidationError("Video resolution is outside the configured supported range.")
    if fps <= 0 or fps > limits.max_fps:
        raise VideoValidationError("Video frame rate is invalid or exceeds the configured limit.")
    cap = cv2.VideoCapture(str(file_path)); ok, frame = cap.read(); cap.release()
    if not ok or frame is None:
        raise VideoValidationError("Video stream cannot be decoded by OpenCV.")
    return {"duration_seconds": round(duration, 3), "width": width, "height": height, "fps": round(fps, 3), "codec": codec, "container": suffix[1:], "size_bytes": file_path.stat().st_size}
