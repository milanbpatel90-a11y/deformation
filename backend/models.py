"""Shared data models for the template deformation pipeline."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FrameShape(str, Enum):
    ROUND = "round"
    SQUARE = "square"
    GEOMETRIC = "geometric"
    AVIATOR = "aviator"
    CAT_EYE = "cat_eye"
    RECTANGLE = "rectangle"
    BROWLINE = "browline"
    RIMLESS = "rimless"


class FrameMaterial(str, Enum):
    METAL = "metal"
    ACETATE = "acetate"
    PLASTIC = "plastic"
    TITANIUM = "titanium"
    MIXED = "mixed"


class FrameFamily(str, Enum):
    GEOMETRIC = "geometric"
    WAYFARER = "wayfarer"
    AVIATOR = "aviator"
    ROUND = "round"
    CAT_EYE = "cat_eye"
    BROWLINE = "browline"
    RIMLESS = "rimless"
    OVERSIZED = "oversized"
    RECTANGLE = "rectangle"
    SQUARE = "square"


class RimType(str, Enum):
    FULL_RIM = "full_rim"
    SEMI_RIMLESS = "semi_rimless"
    RIMLESS = "rimless"


class BridgeType(str, Enum):
    PAD = "pad"
    SADDLE = "saddle"
    KEYHOLE = "keyhole"
    DOUBLE = "double"
    STRAIGHT = "straight"


class Measurements(BaseModel):
    # Broad physical bounds reject malformed values without over-constraining
    # uncommon adult eyewear sizes. All dimensions are millimetres.
    frame_width: float = Field(..., gt=80, le=220, allow_inf_nan=False, description="Total frame width in mm")
    lens_width: float = Field(..., gt=20, le=100, allow_inf_nan=False, description="Single lens width in mm")
    lens_height: float = Field(..., gt=15, le=100, allow_inf_nan=False, description="Single lens height in mm")
    bridge_width: float = Field(..., gt=5, le=50, allow_inf_nan=False, description="Bridge width in mm")
    temple_length: float = Field(..., gt=80, le=220, allow_inf_nan=False, description="Temple arm length in mm")
    rim_thickness: float = Field(default=1.2, ge=0, le=15, allow_inf_nan=False, description="Rim thickness in mm")
    material: FrameMaterial = FrameMaterial.METAL
    shape: FrameShape = FrameShape.GEOMETRIC
    nose_pads: bool = True
    temple_curve_angle: float = Field(default=28.0, ge=0, le=90, allow_inf_nan=False, description="Temple curve in degrees")
    nose_pad_distance: float | None = Field(default=None, ge=0, le=80, allow_inf_nan=False)
    nose_pad_angle: float | None = Field(default=None, ge=0, le=90, allow_inf_nan=False)
    nose_pad_height: float | None = Field(default=None, ge=0, le=25, allow_inf_nan=False)
    color: str = Field(default="#d9a7a2", pattern=r"^#[0-9a-fA-F]{6}$")
    lens_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$", description="Hex color code for the lenses")
    lens_opacity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, description="Opacity of the lenses (0.0 to 1.0)")
    scale_source: Literal["image_estimate", "user_supplied", "manufacturer_spec"] = "image_estimate"
    scale_calibrated: bool = False


class TemplateDimensions(BaseModel):
    frame_width: float
    lens_width: float
    lens_height: float
    bridge_width: float
    temple_length: float
    rim_thickness: float = 1.2
    temple_curve_angle: float = 28.0
    nose_pad_distance: float = 18.0
    nose_pad_angle: float = 15.0
    nose_pad_height: float = 3.0

    @property
    def lens_aspect_ratio(self) -> float:
        return self.lens_width / max(self.lens_height, 1e-6)


class TemplateProfile(BaseModel):
    frame_family: FrameFamily
    material: FrameMaterial
    rim_type: RimType
    bridge_type: BridgeType
    lens_aspect_ratio: float
    source_glb: str | None = None
    fallback_glb: str | None = None
    tags: list[str] = Field(default_factory=list)


class TemplateInfo(BaseModel):
    name: str
    shape: FrameShape
    material: FrameMaterial
    glb_path: str
    dimensions: TemplateDimensions
    parts: list[str]
    profile: TemplateProfile


class StyleClassification(BaseModel):
    shape: FrameShape
    material: FrameMaterial
    nose_pads: bool
    frame_family: FrameFamily
    rim_type: RimType
    bridge_type: BridgeType
    lens_aspect_ratio: float
    confidence: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)


class ScaleFactors(BaseModel):
    frame_x: float = 1.0
    lens_x: float = 1.0
    lens_y: float = 1.0
    bridge_x: float = 1.0
    temple_length: float = 1.0
    temple_thickness: float = 1.0
    rim_thickness: float = 1.0

    @classmethod
    def from_measurements(
        cls, target: Measurements, template: TemplateDimensions
    ) -> ScaleFactors:
        return cls(
            frame_x=target.frame_width / template.frame_width,
            lens_x=target.lens_width / template.lens_width,
            lens_y=target.lens_height / template.lens_height,
            bridge_x=target.bridge_width / template.bridge_width,
            temple_length=target.temple_length / template.temple_length,
            rim_thickness=target.rim_thickness / template.rim_thickness,
        )


class LensContour(BaseModel):
    """Approximated lens contour as normalized polygon vertices."""

    left: list[list[float]] = Field(default_factory=list)
    right: list[list[float]] = Field(default_factory=list)


class ExportMetadata(BaseModel):
    shape: str
    material: str
    frame_width: float
    bridge_width: float
    temple_length: float
    template_used: str
    color: str
    lens_color: str | None = None
    lens_opacity: float | None = None
    scale_source: Literal["image_estimate", "user_supplied", "manufacturer_spec"] = "image_estimate"
    scale_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PipelineStage(BaseModel):
    """Timing and result snapshot for a single pipeline stage."""

    name: str
    status: str = "pending"   # pending | running | done | error
    duration_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)

    def start(self) -> float:
        self.status = "running"
        return time.perf_counter()

    def finish(self, t0: float, **detail: Any) -> None:
        self.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.status = "done"
        self.detail.update(detail)

    def fail(self, t0: float, error: str) -> None:
        self.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.status = "error"
        self.detail["error"] = error


class PipelineReport(BaseModel):
    """Ordered list of stages produced by one pipeline run."""

    stages: list[PipelineStage] = Field(default_factory=list)

    def add(self, name: str) -> PipelineStage:
        stage = PipelineStage(name=name)
        self.stages.append(stage)
        return stage

    def to_dict(self) -> list[dict[str, Any]]:
        return [s.model_dump() for s in self.stages]
