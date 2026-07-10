from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .geometry import MeshRepairReport, repair_mesh
from .utils import get_logger

try:
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RepairBatchReport:
    """Per-object repair summary for the current import."""

    reports: dict[str, MeshRepairReport] = field(default_factory=dict)


def repair_objects(objects: Iterable[Object]) -> RepairBatchReport:
    """Repairs every mesh object in the provided collection."""
    results: dict[str, MeshRepairReport] = {}
    for object_ in objects:
        if getattr(object_, "type", None) != "MESH":
            continue
        results[object_.name] = repair_mesh(object_)
    return RepairBatchReport(reports=results)

