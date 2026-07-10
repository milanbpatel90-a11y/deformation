from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .descriptor_generator import TemplateDescriptor
from .feature_detector import TemplateAnalysis
from .template_metadata import TemplateMetadata
from .validator import ValidationResult


@dataclass(slots=True)
class MetadataArtifacts:
    """Links generated metadata to its descriptor output."""

    metadata: TemplateMetadata
    descriptor: TemplateDescriptor


def generate_metadata(
    template_id: str,
    analysis: TemplateAnalysis,
    validation: ValidationResult,
    descriptor: TemplateDescriptor,
    has_vertex_groups: bool,
    has_uv: bool,
) -> MetadataArtifacts:
    """Builds template metadata from analysis, validation, and descriptor data."""
    metadata = TemplateMetadata(
        id=template_id,
        family=analysis.frame_family,
        material=analysis.material,
        rim=analysis.rim_type,
        bridge=analysis.bridge_type,
        frame_width=analysis.dimensions["frame_width"],
        lens_width=analysis.dimensions["lens_width"],
        lens_height=analysis.dimensions["lens_height"],
        bridge_width=analysis.dimensions["bridge_width"],
        temple_length=analysis.dimensions["temple_length"],
        triangle_count=analysis.triangle_count,
        vertex_count=analysis.vertex_count,
        vertex_groups=has_vertex_groups,
        uv=has_uv,
        validated=validation.status != "FAIL",
        quality_score=analysis.quality_score,
        style=analysis.style,
        frame_height=analysis.dimensions["frame_height"],
        frame_thickness=analysis.dimensions["frame_thickness"],
        validation_score=validation.score,
    )
    return MetadataArtifacts(metadata=metadata, descriptor=descriptor)


def write_metadata(metadata: TemplateMetadata, path: str | Path) -> Path:
    """Writes metadata JSON to disk."""
    return metadata.write_json(path)

