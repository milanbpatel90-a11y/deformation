from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .config import DEFAULT_CONFIG
from .transform import (
    center_origin,
    freeze_transform,
    move_to_world_origin,
    normalize_scale_to_width,
    rotate_to_front_view,
)
from .utils import get_logger

try:
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class NormalizeBatchReport:
    """Captures width normalization factors per object."""

    scale_factors: dict[str, float] = field(default_factory=dict)


def normalize_objects(
    objects: Iterable[Object],
    target_width_mm: float = DEFAULT_CONFIG.validation.default_target_width_mm,
) -> NormalizeBatchReport:
    """Normalizes imported eyewear meshes to a shared transform convention."""
    scale_factors: dict[str, float] = {}
    for object_ in objects:
        if getattr(object_, "type", None) != "MESH":
            continue
        center_origin(object_)
        move_to_world_origin(object_)
        rotate_to_front_view(object_)
        scale_factors[object_.name] = normalize_scale_to_width(object_, target_width_mm)
        freeze_transform(object_)
    return NormalizeBatchReport(scale_factors=scale_factors)

