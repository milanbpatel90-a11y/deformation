from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .utils import _require_blender, get_logger

try:
    import bmesh
    import bpy
    from bpy.types import Object
    from mathutils import Vector
except ImportError:  # pragma: no cover - Blender runtime only
    bmesh = None
    bpy = None
    Object = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class UVReport:
    """Detailed UV validation and processing report."""

    valid: bool
    has_uv: bool
    overlapping_faces: int = 0
    flipped_faces: int = 0
    missing_faces: int = 0
    uv_layer_name: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _mesh_objects(objects: Optional[Iterable[Object]] = None) -> list[Object]:
    """Returns mesh objects from input or scene."""
    _require_blender()
    source = list(objects) if objects is not None else list(bpy.context.scene.objects)
    return [object_ for object_ in source if object_.type == "MESH"]


def _uv_report_for_object(object_: Object) -> UVReport:
    """Builds a UV report for a single object."""
    mesh = object_.data
    has_uv = bool(mesh.uv_layers)
    report = UVReport(valid=has_uv, has_uv=has_uv)
    if not has_uv:
        report.errors.append(f"Object '{object_.name}' has no UV map.")
        report.missing_faces = len(mesh.polygons)
        return report

    uv_layer = mesh.uv_layers.active
    report.uv_layer_name = uv_layer.name
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        uv = bm.loops.layers.uv.active
        if uv is None:
            report.valid = False
            report.errors.append(f"Object '{object_.name}' has an invalid UV layer.")
            return report

        seen_boxes: list[tuple[float, float, float, float]] = []
        for face in bm.faces:
            uv_coords = [loop[uv].uv.copy() for loop in face.loops]
            if not uv_coords:
                report.missing_faces += 1
                continue
            min_u = min(coord.x for coord in uv_coords)
            min_v = min(coord.y for coord in uv_coords)
            max_u = max(coord.x for coord in uv_coords)
            max_v = max(coord.y for coord in uv_coords)
            if max_u == min_u or max_v == min_v:
                report.missing_faces += 1
            signed_area = 0.0
            for index, coord in enumerate(uv_coords):
                next_coord = uv_coords[(index + 1) % len(uv_coords)]
                signed_area += coord.x * next_coord.y - next_coord.x * coord.y
            if signed_area < 0.0:
                report.flipped_faces += 1
            current_box = (min_u, min_v, max_u, max_v)
            for other_box in seen_boxes:
                if not (
                    current_box[2] <= other_box[0]
                    or current_box[0] >= other_box[2]
                    or current_box[3] <= other_box[1]
                    or current_box[1] >= other_box[3]
                ):
                    report.overlapping_faces += 1
                    break
            seen_boxes.append(current_box)
        report.valid = report.missing_faces == 0
        return report
    finally:
        bm.free()


def validate_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, UVReport]:
    """Validates UV data for one or more mesh objects."""
    return {object_.name: _uv_report_for_object(object_) for object_ in _mesh_objects(objects)}


def detect_missing_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, bool]:
    """Detects objects without usable UV maps."""
    return {name: not report.has_uv for name, report in validate_uv(objects).items()}


def detect_overlapping_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, int]:
    """Detects approximate UV overlaps per object."""
    return {name: report.overlapping_faces for name, report in validate_uv(objects).items()}


def preserve_existing_uv(object_: Object) -> str:
    """Ensures an object's current UV layer remains active."""
    _require_blender()
    if not object_.data.uv_layers:
        return ""
    if object_.data.uv_layers.active is None:
        object_.data.uv_layers.active = object_.data.uv_layers[0]
    return object_.data.uv_layers.active.name


def generate_uv(object_: Object, method: str = "SMART_PROJECT") -> UVReport:
    """Generates UVs for a mesh object using Blender unwrap operators."""
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("UV generation requires a mesh object.")
    preserve_existing_uv(object_)
    bpy.context.view_layer.objects.active = object_
    object_.select_set(True)
    previous_mode = object_.mode
    try:
        if not object_.data.uv_layers:
            object_.data.uv_layers.new(name="UVMap")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if method.upper() == "ANGLE_BASED":
            bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.003)
        else:
            bpy.ops.uv.smart_project(angle_limit=1.1519, island_margin=0.003)
        bpy.ops.object.mode_set(mode="OBJECT")
        return _uv_report_for_object(object_)
    finally:
        try:
            bpy.ops.object.mode_set(mode=previous_mode)
        except Exception:
            bpy.ops.object.mode_set(mode="OBJECT")


def pack_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, UVReport]:
    """Packs UV islands for one or more objects."""
    reports: dict[str, UVReport] = {}
    for object_ in _mesh_objects(objects):
        bpy.context.view_layer.objects.active = object_
        object_.select_set(True)
        previous_mode = object_.mode
        try:
            if not object_.data.uv_layers:
                reports[object_.name] = generate_uv(object_)
                continue
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
            bpy.ops.object.mode_set(mode="OBJECT")
            reports[object_.name] = _uv_report_for_object(object_)
        finally:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                bpy.ops.object.mode_set(mode="OBJECT")
    return reports


def flip_incorrect_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, UVReport]:
    """Flips UVs for faces with negative signed UV area."""
    reports: dict[str, UVReport] = {}
    for object_ in _mesh_objects(objects):
        mesh = object_.data
        if not mesh.uv_layers:
            reports[object_.name] = _uv_report_for_object(object_)
            continue
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            uv = bm.loops.layers.uv.active
            flipped = 0
            for face in bm.faces:
                coords = [loop[uv].uv.copy() for loop in face.loops]
                signed_area = 0.0
                for index, coord in enumerate(coords):
                    next_coord = coords[(index + 1) % len(coords)]
                    signed_area += coord.x * next_coord.y - next_coord.x * coord.y
                if signed_area < 0.0:
                    reversed_coords = list(reversed(coords))
                    for loop, coord in zip(face.loops, reversed_coords):
                        loop[uv].uv = coord
                    flipped += 1
            bm.to_mesh(mesh)
            mesh.update()
            report = _uv_report_for_object(object_)
            report.flipped_faces = flipped
            reports[object_.name] = report
        finally:
            bm.free()
    return reports


def repair_uv(objects: Optional[Iterable[Object]] = None) -> dict[str, UVReport]:
    """Repairs UVs by generating, flipping, and packing as needed."""
    reports: dict[str, UVReport] = {}
    for object_ in _mesh_objects(objects):
        report = _uv_report_for_object(object_)
        if not report.has_uv or report.missing_faces > 0:
            report = generate_uv(object_)
        if report.flipped_faces > 0:
            report = flip_incorrect_uv([object_])[object_.name]
        pack_result = pack_uv([object_])[object_.name]
        reports[object_.name] = pack_result
    return reports


__all__ = [
    "UVReport",
    "detect_missing_uv",
    "detect_overlapping_uv",
    "flip_incorrect_uv",
    "generate_uv",
    "pack_uv",
    "preserve_existing_uv",
    "repair_uv",
    "validate_uv",
]
