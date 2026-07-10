from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .mesh_splitter import (
    BRIDGE,
    FRAME,
    LEFT_LENS,
    LEFT_PAD,
    LEFT_TEMPLE,
    RIGHT_LENS,
    RIGHT_PAD,
    RIGHT_TEMPLE,
    split_eyewear_mesh,
)
from .utils import _require_blender, get_logger

try:
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)

GROUP_NAME_MAP = {
    FRAME: "VG_FRAME",
    "LeftRim": "VG_LEFT_RIM",
    "RightRim": "VG_RIGHT_RIM",
    BRIDGE: "VG_BRIDGE",
    LEFT_LENS: "VG_LEFT_LENS",
    RIGHT_LENS: "VG_RIGHT_LENS",
    LEFT_TEMPLE: "VG_LEFT_TEMPLE",
    RIGHT_TEMPLE: "VG_RIGHT_TEMPLE",
    LEFT_PAD: "VG_LEFT_PAD",
    RIGHT_PAD: "VG_RIGHT_PAD",
}


@dataclass(slots=True)
class VertexGroupBuildReport:
    """Summarizes generated vertex groups."""

    groups_created: list[str] = field(default_factory=list)
    groups_updated: list[str] = field(default_factory=list)
    vertex_counts: dict[str, int] = field(default_factory=dict)


def _mesh_objects(objects: Optional[Iterable[Object]] = None) -> list[Object]:
    """Returns mesh objects from input or scene."""
    _require_blender()
    source = list(objects) if objects is not None else list(__import__("bpy").context.scene.objects)
    return [object_ for object_ in source if object_.type == "MESH"]


def _ensure_group(object_: Object, name: str):
    """Returns an object vertex group, creating it when needed."""
    group = object_.vertex_groups.get(name)
    if group is None:
        group = object_.vertex_groups.new(name=name)
    return group


def _clear_group_weights(object_: Object, group_name: str) -> None:
    """Clears all weights from a group when it exists."""
    group = object_.vertex_groups.get(group_name)
    if group is None:
        return
    indices = [vertex.index for vertex in object_.data.vertices]
    if indices:
        group.remove(indices)


def _assign_all_vertices(object_: Object, group_name: str) -> int:
    """Assigns every vertex in an object to a group."""
    group = _ensure_group(object_, group_name)
    indices = [vertex.index for vertex in object_.data.vertices]
    if indices:
        group.add(indices, 1.0, "REPLACE")
    return len(indices)


def _assign_side_vertices(
    object_: Object,
    left_group_name: str,
    right_group_name: str,
) -> tuple[int, int]:
    """Assigns vertices to left and right groups based on world X position."""
    left_group = _ensure_group(object_, left_group_name)
    right_group = _ensure_group(object_, right_group_name)
    left_indices: list[int] = []
    right_indices: list[int] = []
    for vertex in object_.data.vertices:
        world_co = object_.matrix_world @ vertex.co
        if world_co.x < 0.0:
            left_indices.append(vertex.index)
        else:
            right_indices.append(vertex.index)
    if left_indices:
        left_group.add(left_indices, 1.0, "REPLACE")
    if right_indices:
        right_group.add(right_indices, 1.0, "REPLACE")
    return len(left_indices), len(right_indices)


def build_vertex_groups(objects: Optional[Iterable[Object]] = None) -> VertexGroupBuildReport:
    """Automatically builds deformation vertex groups for eyewear templates."""
    mesh_objects = _mesh_objects(objects)
    parts = split_eyewear_mesh(mesh_objects, split_components=False)
    report = VertexGroupBuildReport()

    for object_ in mesh_objects:
        for group_name in GROUP_NAME_MAP.values():
            _clear_group_weights(object_, group_name)

    for part_name, part in parts.items():
        if part_name in {FRAME, BRIDGE, LEFT_LENS, RIGHT_LENS, LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_PAD, RIGHT_PAD}:
            for object_ in part.objects:
                count = _assign_all_vertices(object_, GROUP_NAME_MAP[part_name])
                report.vertex_counts[GROUP_NAME_MAP[part_name]] = report.vertex_counts.get(GROUP_NAME_MAP[part_name], 0) + count
                report.groups_updated.append(GROUP_NAME_MAP[part_name])

    for object_ in parts[FRAME].objects:
        left_count, right_count = _assign_side_vertices(
            object_,
            GROUP_NAME_MAP["LeftRim"],
            GROUP_NAME_MAP["RightRim"],
        )
        report.vertex_counts[GROUP_NAME_MAP["LeftRim"]] = report.vertex_counts.get(GROUP_NAME_MAP["LeftRim"], 0) + left_count
        report.vertex_counts[GROUP_NAME_MAP["RightRim"]] = report.vertex_counts.get(GROUP_NAME_MAP["RightRim"], 0) + right_count
        report.groups_updated.extend([GROUP_NAME_MAP["LeftRim"], GROUP_NAME_MAP["RightRim"]])

    unique_groups = sorted(set(report.groups_updated))
    report.groups_created = unique_groups
    report.groups_updated = unique_groups
    return report


__all__ = ["GROUP_NAME_MAP", "VertexGroupBuildReport", "build_vertex_groups"]
