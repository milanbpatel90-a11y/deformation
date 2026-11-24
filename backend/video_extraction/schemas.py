"""Data models for the video-to-reference extraction feature."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ViewLabel(str, Enum):
    """Viewing angle buckets. See the "View classification" section of the
    package README: this is a heuristic classifier layered on segmentation
    geometry, not a calibrated pose estimator."""

    FRONT = "front"
    FRONT_LEFT_45 = "front_left_45"
    FRONT_RIGHT_45 = "front_right_45"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    TOP = "top"
    UNKNOWN = "unknown"


class CalibrationStatus(str, Enum):
    """How trustworthy a measurement's millimetre value is.

    - calibrated: derived using a merchant-supplied known physical dimension
      (or, in the future, a detected reference marker).
    - estimated: derived from the existing single/multi-image measurement
      heuristics, which assume a typical product-photo framing. Treat as
      approximate, not metrology-grade.
    - pixel: no mm scale could be established at all; the value is in pixels
      and MUST NOT be interpreted as millimetres by any downstream consumer.
    """

    CALIBRATED = "calibrated"
    ESTIMATED = "estimated"
    PIXEL = "pixel"


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    """Probed properties of an uploaded video."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec_fourcc: str | None = None


class QualityMetrics(BaseModel):
    """Per-frame quality signals, all resolution-normalized."""

    sharpness_score: float
    is_blurry: bool
    dark_fraction: float
    bright_fraction: float
    exposure_ok: bool
    glare_fraction: float
    glare_ok: bool
    overall_quality: float = Field(description="0-100 composite quality score")
    reasons: list[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    """Glasses detection/segmentation summary for one frame."""

    detected: bool
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in pixels
    area_fraction: float = 0.0
    class_confidences: dict[str, float] = Field(default_factory=dict)
    missing_components: list[str] = Field(default_factory=list)
    model_type: str = "unknown"


class FrameCandidate(BaseModel):
    """A sampled video frame considered for selection."""

    frame_index: int
    timestamp_sec: float
    view: ViewLabel = ViewLabel.UNKNOWN
    view_confidence: float = 0.0
    quality: QualityMetrics
    detection: DetectionResult
    phash: str | None = None
    composite_score: float = 0.0
    rejected: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class SelectedFrame(BaseModel):
    """A frame chosen as a reference image, with its saved asset paths."""

    frame_index: int
    timestamp_sec: float
    view: ViewLabel
    view_confidence: float
    composite_score: float
    image_path: str
    overlay_path: str | None = None
    cutout_path: str | None = None
    mask_paths: dict[str, str] = Field(default_factory=dict)
    detection: DetectionResult
    manually_replaced: bool = False
    approved: bool = False


class MeasurementValue(BaseModel):
    """One measured quantity with full provenance."""

    name: str
    value: float
    unit: str
    confidence: float
    source_view: str
    calibration_status: CalibrationStatus


class MeasurementsBundle(BaseModel):
    """Fused measurements ready (or not) for the deformation pipeline."""

    fused: dict[str, Any] | None = None
    per_measurement: list[MeasurementValue] = Field(default_factory=list)
    calibration_status: CalibrationStatus
    calibration_reference_mm: float | None = None
    warnings: list[str] = Field(default_factory=list)
    sufficient_for_deformation: bool = False


class ExtractionReport(BaseModel):
    """Top-level report.json written for one job."""

    job_id: str
    video_metadata: VideoMetadata
    candidates_sampled: int
    candidates_rejected: int
    selected_views: list[str]
    missing_views: list[str]
    warnings: list[str] = Field(default_factory=list)
    measurement_warnings: list[str] = Field(default_factory=list)
    calibration_status: CalibrationStatus


class CalibrationInput(BaseModel):
    """Optional merchant-supplied calibration for absolute measurements."""

    known_frame_width_mm: float | None = Field(
        default=None, ge=80.0, le=200.0,
        description="Merchant-supplied true total frame width in millimetres.",
    )
