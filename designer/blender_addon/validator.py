from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .config import DEFAULT_CONFIG
from .geometry import (
    calculate_bounding_box,
    calculate_dimensions_mm,
    check_non_manifold,
    check_self_intersections,
)
from .materials import validate_materials
from .utils import (
    _require_blender,
    count_triangles,
    count_vertices,
    get_logger,
    get_mesh_objects,
    has_uv,
    has_vertex_groups,
)

try:
    import bmesh
    import bpy
    from bpy.types import Collection, Object
    from mathutils import Quaternion
except ImportError:  # pragma: no cover - Blender runtime only
    bmesh = None
    bpy = None
    Collection = object  # type: ignore[assignment]
    Object = object  # type: ignore[assignment]
    Quaternion = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """Aggregates validation feedback for template assets."""

    status: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    score: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


def _collect_objects(
    objects: Optional[Iterable[Object]] = None,
    collection: Optional[Collection] = None,
) -> list[Object]:
    """Collects validation target objects."""
    _require_blender()
    if objects is not None:
        return list(objects)
    if collection is not None:
        return list(collection.objects)
    return list(bpy.context.scene.objects)


def _is_name_clean(name: str) -> bool:
    """Checks whether an object name is export-safe."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    return bool(name) and all(character in allowed for character in name)


def _rotation_is_identity(object_: Object, epsilon: float = 1e-4) -> bool:
    """Checks whether object rotation is effectively zero."""
    if object_.rotation_mode == "QUATERNION":
        return (
            object_.rotation_quaternion - Quaternion((1.0, 0.0, 0.0, 0.0))
        ).magnitude <= epsilon
    if object_.rotation_mode == "AXIS_ANGLE":
        angle, *_ = object_.rotation_axis_angle
        return abs(float(angle)) <= epsilon
    return all(abs(float(value)) <= epsilon for value in object_.rotation_euler)


def _scale_is_identity(object_: Object, epsilon: float = 1e-4) -> bool:
    """Checks whether object scale is effectively one."""
    return all(abs(float(value) - 1.0) <= epsilon for value in object_.scale)


def _origin_is_world_origin(object_: Object, epsilon_m: float) -> bool:
    """Checks whether object origin is at the world origin."""
    return object_.location.length <= epsilon_m


def _duplicate_vertex_count(object_: Object) -> int:
    """Counts duplicate vertices using a copy of mesh data."""
    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        before_count = len(bm.verts)
        bmesh.ops.remove_doubles(
            bm,
            verts=list(bm.verts),
            dist=DEFAULT_CONFIG.validation.merge_distance_mm / 1000.0,
        )
        return before_count - len(bm.verts)
    finally:
        bm.free()


def _duplicate_face_count(object_: Object) -> int:
    """Counts duplicate faces using rounded vertex signatures."""
    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        signatures: set[tuple[tuple[float, float, float], ...]] = set()
        duplicates = 0
        for face in bm.faces:
            signature = tuple(
                sorted(
                    (
                        round(float(vert.co.x), 6),
                        round(float(vert.co.y), 6),
                        round(float(vert.co.z), 6),
                    )
                    for vert in face.verts
                )
            )
            if signature in signatures:
                duplicates += 1
            else:
                signatures.add(signature)
        return duplicates
    finally:
        bm.free()


def validate_template(
    objects: Optional[Iterable[Object]] = None,
    collection: Optional[Collection] = None,
) -> ValidationResult:
    """Validates a Blender eyewear template collection."""
    _require_blender()
    validation = DEFAULT_CONFIG.validation
    all_objects = _collect_objects(objects=objects, collection=collection)
    mesh_objects = [object_ for object_ in all_objects if object_.type == "MESH"]
    empty_objects = [object_ for object_ in all_objects if object_.type == "EMPTY"]
    hidden_objects = [object_ for object_ in all_objects if object_.hide_get()]
    animated_objects = [
        object_
        for object_ in all_objects
        if object_.animation_data is not None and object_.animation_data.action is not None
    ]
    shape_key_objects = [
        object_
        for object_ in mesh_objects
        if object_.data.shape_keys is not None and object_.data.shape_keys.key_blocks
    ]

    warnings: list[str] = []
    errors: list[str] = []
    recommendations: set[str] = set()
    metrics: dict[str, Any] = {}
    score = 100

    metrics["object_count"] = len(all_objects)
    metrics["mesh_count"] = len(mesh_objects)
    metrics["empty_object_count"] = len(empty_objects)
    metrics["hidden_object_count"] = len(hidden_objects)
    metrics["animated_object_count"] = len(animated_objects)
    metrics["shape_key_object_count"] = len(shape_key_objects)

    if not mesh_objects:
        errors.append("No mesh objects found for validation.")
        score -= 50

    if empty_objects:
        warnings.append(f"Found {len(empty_objects)} empty helper objects.")
        recommendations.add("Remove empty helper objects before template export.")
        score -= min(5 * len(empty_objects), 10)

    if hidden_objects:
        warnings.append(f"Found {len(hidden_objects)} hidden objects.")
        recommendations.add("Unhide or remove hidden objects before export.")
        score -= min(3 * len(hidden_objects), 12)

    if animated_objects:
        warnings.append(f"Found {len(animated_objects)} animated objects.")
        recommendations.add("Bake or remove animations from template assets.")
        score -= min(5 * len(animated_objects), 15)

    if shape_key_objects:
        warnings.append(f"Found {len(shape_key_objects)} objects with shape keys.")
        recommendations.add("Remove shape keys from static template geometry.")
        score -= min(5 * len(shape_key_objects), 15)

    unique_names = {object_.name for object_ in all_objects}
    if len(unique_names) != len(all_objects):
        errors.append("Object names are not unique.")
        recommendations.add("Rename objects so every object name is unique.")
        score -= 10

    invalid_names = [object_.name for object_ in all_objects if not _is_name_clean(object_.name)]
    if invalid_names:
        warnings.append(f"Found export-unsafe object names: {', '.join(sorted(invalid_names))}.")
        recommendations.add("Use ASCII-safe object names with letters, digits, '_', '-', or '.'.")
        score -= min(2 * len(invalid_names), 10)

    epsilon_m = validation.bounding_box_epsilon_mm / 1000.0
    mesh_metrics: list[dict[str, Any]] = []
    total_triangles = 0
    total_vertices = 0
    total_material_slots = 0

    for object_ in mesh_objects:
        object_metrics: dict[str, Any] = {"name": object_.name}
        object_metrics["origin_ok"] = _origin_is_world_origin(object_, epsilon_m)
        object_metrics["rotation_ok"] = _rotation_is_identity(object_)
        object_metrics["scale_ok"] = _scale_is_identity(object_)
        object_metrics["has_uv"] = has_uv(object_)
        object_metrics["has_vertex_groups"] = has_vertex_groups(object_)
        object_metrics["triangle_count"] = count_triangles(object_)
        object_metrics["vertex_count"] = count_vertices(object_)
        object_metrics["material_count"] = len([slot for slot in object_.material_slots if slot.material])
        object_metrics["duplicate_vertices"] = _duplicate_vertex_count(object_)
        object_metrics["duplicate_faces"] = _duplicate_face_count(object_)
        object_metrics["non_manifold"] = check_non_manifold(object_)
        object_metrics["self_intersections"] = check_self_intersections(object_)
        object_metrics["bounding_box"] = calculate_bounding_box(object_)
        object_metrics["dimensions_mm"] = calculate_dimensions_mm(object_)
        object_metrics["has_custom_normals"] = bool(getattr(object_.data, "has_custom_normals", False))

        total_triangles += object_metrics["triangle_count"]
        total_vertices += object_metrics["vertex_count"]
        total_material_slots += object_metrics["material_count"]

        if not object_metrics["origin_ok"]:
            warnings.append(f"Object '{object_.name}' is not at the world origin.")
            recommendations.add("Move object origins to world origin before export.")
            score -= 3

        if not object_metrics["rotation_ok"]:
            warnings.append(f"Object '{object_.name}' has unapplied rotation.")
            recommendations.add("Apply object rotations before final export.")
            score -= 3

        if not object_metrics["scale_ok"]:
            warnings.append(f"Object '{object_.name}' has unapplied scale.")
            recommendations.add("Apply object scales before final export.")
            score -= 3

        if validation.require_uvs and not object_metrics["has_uv"]:
            errors.append(f"Object '{object_.name}' has no UV map.")
            recommendations.add("Create UVs for all exported template meshes.")
            score -= 10

        if validation.require_vertex_groups and not object_metrics["has_vertex_groups"]:
            warnings.append(f"Object '{object_.name}' has no vertex groups.")
            recommendations.add("Add template deformation vertex groups where required.")
            score -= 4

        if object_metrics["triangle_count"] > validation.max_triangle_count:
            errors.append(
                f"Object '{object_.name}' exceeds triangle limit "
                f"({object_metrics['triangle_count']} > {validation.max_triangle_count})."
            )
            recommendations.add("Reduce triangle density to stay within budget.")
            score -= 10

        if object_metrics["vertex_count"] < validation.min_vertex_count:
            warnings.append(
                f"Object '{object_.name}' has very low vertex density "
                f"({object_metrics['vertex_count']})."
            )
            score -= 3

        if object_metrics["vertex_count"] > validation.max_vertex_count:
            errors.append(
                f"Object '{object_.name}' exceeds vertex limit "
                f"({object_metrics['vertex_count']} > {validation.max_vertex_count})."
            )
            recommendations.add("Reduce vertex density to stay within budget.")
            score -= 10

        if object_metrics["duplicate_vertices"] > 0:
            warnings.append(
                f"Object '{object_.name}' contains {object_metrics['duplicate_vertices']} duplicate vertices."
            )
            recommendations.add("Merge duplicate vertices before export.")
            score -= min(object_metrics["duplicate_vertices"], 8)

        if object_metrics["duplicate_faces"] > 0:
            warnings.append(
                f"Object '{object_.name}' contains {object_metrics['duplicate_faces']} duplicate faces."
            )
            recommendations.add("Remove duplicate faces before export.")
            score -= min(object_metrics["duplicate_faces"], 8)

        non_manifold_report = object_metrics["non_manifold"]
        if non_manifold_report.is_non_manifold:
            errors.append(
                f"Object '{object_.name}' is non-manifold "
                f"({non_manifold_report.non_manifold_edges} edges, "
                f"{non_manifold_report.non_manifold_vertices} vertices)."
            )
            recommendations.add("Repair non-manifold edges and vertices.")
            score -= min(
                int(non_manifold_report.non_manifold_ratio * 100) + 5,
                20,
            )

        self_intersection_report = object_metrics["self_intersections"]
        if self_intersection_report.has_self_intersections:
            errors.append(
                f"Object '{object_.name}' has {self_intersection_report.intersecting_pairs} self-intersections."
            )
            recommendations.add("Resolve self-intersections in the mesh shell.")
            score -= min(self_intersection_report.intersecting_pairs * 2, 20)

        width_mm = float(object_metrics["dimensions_mm"][0])
        if width_mm < validation.min_width_mm or width_mm > validation.max_width_mm:
            warnings.append(
                f"Object '{object_.name}' width {width_mm:.2f} mm is outside "
                f"expected range {validation.min_width_mm:.2f}-{validation.max_width_mm:.2f} mm."
            )
            recommendations.add("Normalize mesh width to the expected template range.")
            score -= 6

        bbox = object_metrics["bounding_box"]
        if min(float(bbox.size.x), float(bbox.size.y), float(bbox.size.z)) <= 0.0:
            errors.append(f"Object '{object_.name}' has an invalid bounding box.")
            recommendations.add("Repair invalid bounding boxes before export.")
            score -= 10

        if object_metrics["material_count"] == 0:
            warnings.append(f"Object '{object_.name}' has no assigned material.")
            recommendations.add("Assign at least one PBR material to each mesh.")
            score -= 4

        if object_metrics["has_custom_normals"]:
            warnings.append(f"Object '{object_.name}' contains custom normals.")
            recommendations.add("Clean custom normals unless explicitly required.")
            score -= 2

        mesh_metrics.append(object_metrics)

    material_report = validate_materials(mesh_objects)
    metrics["materials"] = {
        "material_count": material_report.material_count,
        "principled_count": material_report.principled_count,
        "duplicate_count": material_report.duplicate_count,
        "missing_texture_count": material_report.missing_texture_count,
    }
    metrics["mesh_metrics"] = mesh_metrics
    metrics["total_triangles"] = total_triangles
    metrics["total_vertices"] = total_vertices
    metrics["total_material_slots"] = total_material_slots

    warnings.extend(material_report.warnings)
    errors.extend(material_report.errors)
    if material_report.duplicate_count:
        recommendations.add("Merge duplicate materials to simplify GLB exports.")
        score -= min(material_report.duplicate_count * 2, 10)
    if material_report.missing_texture_count:
        recommendations.add("Repair missing texture paths or pack textures into the GLB.")
        score -= min(material_report.missing_texture_count * 2, 10)
    if material_report.errors:
        recommendations.add("Convert all materials to Principled PBR before export.")
        score -= min(len(material_report.errors) * 5, 20)

    score = max(min(score, 100), 0)
    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    return ValidationResult(
        status=status,
        warnings=warnings,
        errors=errors,
        score=score,
        metrics=metrics,
        recommendations=sorted(recommendations),
    )


__all__ = ["ValidationResult", "validate_template"]
