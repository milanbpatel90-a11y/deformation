from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Iterable, List, Optional, Sequence, Tuple
import uuid

from .config import DEFAULT_CONFIG

try:
    import bpy
    import bmesh
    from bpy.types import Material, Object
    from mathutils import Euler, Matrix, Quaternion, Vector
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    bmesh = None
    Material = object  # type: ignore[assignment]
    Object = object  # type: ignore[assignment]
    Euler = object  # type: ignore[assignment]
    Matrix = object  # type: ignore[assignment]
    Quaternion = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]


LOGGER = logging.getLogger(DEFAULT_CONFIG.logging.logger_name)


class BlenderContextError(RuntimeError):
    """Raised when a Blender-specific utility is called outside Blender."""


class BlenderOperationError(RuntimeError):
    """Raised when a Blender operator or mesh operation fails."""


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a configured logger for add-on modules.

    Args:
        name: Optional logger name. When omitted, returns the root add-on logger.

    Returns:
        Configured logger instance.
    """
    _configure_logger(LOGGER)
    if not name or name == LOGGER.name:
        return LOGGER

    logger = logging.getLogger(name)
    if not logger.handlers:
        for handler in LOGGER.handlers:
            logger.addHandler(handler)
    logger.setLevel(LOGGER.level)
    logger.propagate = DEFAULT_CONFIG.logging.propagate
    return logger


def _configure_logger(logger: logging.Logger) -> None:
    """Configures the shared logger once per Python session."""
    if getattr(logger, "_defirmation_configured", False):
        return

    config = DEFAULT_CONFIG.logging
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.propagate = config.propagate

    formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.captureWarnings(config.capture_warnings)
    logger._defirmation_configured = True  # type: ignore[attr-defined]


def list_glb_files(directory: str | Path, recursive: bool = False) -> List[Path]:
    """Lists GLB files within a directory.

    Args:
        directory: Directory to search.
        recursive: When ``True``, search subdirectories recursively.

    Returns:
        Sorted list of GLB file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    pattern = "**/*.glb" if recursive else "*.glb"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def ensure_directory(directory: str | Path) -> Path:
    """Creates a directory if needed and returns its resolved path.

    Args:
        directory: Directory path to create.

    Returns:
        Resolved directory path.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_unique_name(base_name: str, existing_names: Optional[Iterable[str]] = None) -> str:
    """Generates a unique name based on an existing name set.

    Args:
        base_name: Preferred base name.
        existing_names: Existing names to avoid.

    Returns:
        Unique name suitable for Blender data blocks or files.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name.strip()).strip("._") or "item"
    names = set(existing_names or ())
    if sanitized not in names:
        return sanitized

    index = 1
    while f"{sanitized}_{index:03d}" in names:
        index += 1
    return f"{sanitized}_{index:03d}"


def safe_export_path(
    directory: str | Path,
    file_name: str,
    extension: str = ".glb",
    overwrite: bool = False,
) -> Path:
    """Builds a safe export path and avoids accidental file collisions.

    Args:
        directory: Target export directory.
        file_name: Base file name without extension.
        extension: File extension, with or without a leading dot.
        overwrite: When ``True``, return the direct target even if it exists.

    Returns:
        Export path that is safe to write.
    """
    target_dir = ensure_directory(directory)
    suffix = extension if extension.startswith(".") else f".{extension}"
    sanitized_name = generate_unique_name(Path(file_name).stem)
    candidate = target_dir / f"{sanitized_name}{suffix.lower()}"
    if overwrite or not candidate.exists():
        return candidate

    stem = candidate.stem
    index = 1
    while True:
        unique = candidate.with_name(f"{stem}_{index:03d}{suffix}")
        if not unique.exists():
            return unique
        index += 1


def _require_blender() -> None:
    """Ensures Blender Python modules are available."""
    if bpy is None or bmesh is None:
        raise BlenderContextError("This utility requires Blender's Python runtime.")


def _editable_mesh_objects(objects: Iterable[Object]) -> List[Object]:
    """Filters editable mesh objects."""
    return [obj for obj in objects if getattr(obj, "type", None) == "MESH"]


def get_selected_meshes(context: Optional["bpy.types.Context"] = None) -> List[Object]:
    """Returns currently selected mesh objects.

    Args:
        context: Optional Blender context.

    Returns:
        Selected mesh objects.
    """
    _require_blender()
    current_context = context or bpy.context
    return _editable_mesh_objects(current_context.selected_objects)


def get_mesh_objects(
    collection: Optional["bpy.types.Collection"] = None,
    include_hidden: bool = False,
) -> List[Object]:
    """Returns mesh objects from a collection or the active scene.

    Args:
        collection: Optional collection to scan.
        include_hidden: When ``True``, include hidden objects.

    Returns:
        Mesh objects in deterministic name order.
    """
    _require_blender()
    source = collection.objects if collection else bpy.context.scene.objects
    meshes = []
    for obj in source:
        if obj.type != "MESH":
            continue
        if not include_hidden and obj.hide_get():
            continue
        meshes.append(obj)
    return sorted(meshes, key=lambda item: item.name.casefold())


def get_object_bounds(object_: Object, world_space: bool = True) -> Tuple[Vector, Vector]:
    """Calculates object bounding-box minimum and maximum corners.

    Args:
        object_: Mesh object to evaluate.
        world_space: When ``True``, return world-space bounds.

    Returns:
        Tuple of minimum and maximum bounding-box vectors.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError(f"Expected a mesh object, got: {object_.type}")

    corners = [Vector(corner) for corner in object_.bound_box]
    if world_space:
        corners = [object_.matrix_world @ corner for corner in corners]

    min_corner = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_corner = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_corner, max_corner


def calculate_dimensions_mm(object_: Object) -> Tuple[float, float, float]:
    """Calculates axis-aligned object dimensions in millimeters.

    Args:
        object_: Mesh object to evaluate.

    Returns:
        Width, depth, and height in millimeters.
    """
    min_corner, max_corner = get_object_bounds(object_, world_space=True)
    dimensions_m = max_corner - min_corner
    return (
        float(dimensions_m.x * 1000.0),
        float(dimensions_m.y * 1000.0),
        float(dimensions_m.z * 1000.0),
    )


def center_origin(object_: Object) -> None:
    """Moves an object's origin to its geometry bounds center.

    Args:
        object_: Target object.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("center_origin requires a mesh object.")

    local_center = sum((Vector(corner) for corner in object_.bound_box), Vector()) / 8.0
    mesh = object_.data
    translation = -local_center
    mesh.transform(Matrix.Translation(translation))
    object_.matrix_world.translation = object_.matrix_world @ local_center
    mesh.update()


def apply_rotation(object_: Object) -> None:
    """Applies object rotation to mesh data.

    Args:
        object_: Target mesh object.
    """
    _require_blender()
    _apply_transform(object_, location=False, rotation=True, scale=False)


def apply_scale(object_: Object) -> None:
    """Applies object scale to mesh data.

    Args:
        object_: Target mesh object.
    """
    _require_blender()
    _apply_transform(object_, location=False, rotation=False, scale=True)


def _apply_transform(
    object_: Object,
    location: bool,
    rotation: bool,
    scale: bool,
) -> None:
    """Applies object transforms without relying on screen context."""
    if object_.type != "MESH":
        raise TypeError("Transform application requires a mesh object.")

    matrix = object_.matrix_basis.copy()
    loc, rot, scl = matrix.decompose()
    transform_matrix = Matrix.Identity(4)

    if location:
        transform_matrix @= Matrix.Translation(loc)
    if rotation:
        transform_matrix @= rot.to_matrix().to_4x4()
    if scale:
        scale_matrix = Matrix.Diagonal((scl.x, scl.y, scl.z, 1.0))
        transform_matrix @= scale_matrix

    object_.data.transform(transform_matrix)
    new_loc = (0.0, 0.0, 0.0) if location else loc
    new_scale = (1.0, 1.0, 1.0) if scale else scl

    object_.location = new_loc
    if rotation:
        if object_.rotation_mode == "QUATERNION":
            object_.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        elif object_.rotation_mode == "AXIS_ANGLE":
            object_.rotation_axis_angle = (0.0, 0.0, 0.0, 1.0)
        else:
            object_.rotation_euler = Euler((0.0, 0.0, 0.0), object_.rotation_mode)
    if scale:
        object_.scale = new_scale

    object_.data.update()


def merge_duplicate_vertices(
    object_: Object,
    distance_mm: Optional[float] = None,
) -> int:
    """Merges duplicate vertices using Blender's bmesh operators.

    Args:
        object_: Target mesh object.
        distance_mm: Merge distance in millimeters.

    Returns:
        Number of vertices removed.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("merge_duplicate_vertices requires a mesh object.")

    threshold_mm = distance_mm or DEFAULT_CONFIG.validation.merge_distance_mm
    threshold_m = threshold_mm / 1000.0

    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        before_count = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=threshold_m)
        after_count = len(bm.verts)
        bm.to_mesh(object_.data)
        object_.data.update()
        return before_count - after_count
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to merge duplicate vertices on '{object_.name}'."
        ) from exc
    finally:
        bm.free()


def recalculate_normals(object_: Object, inside: bool = False) -> None:
    """Recalculates mesh normals.

    Args:
        object_: Target mesh object.
        inside: When ``True``, normals are oriented inward.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("recalculate_normals requires a mesh object.")

    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        faces = list(bm.faces)
        if faces:
            bmesh.ops.recalc_face_normals(bm, faces=faces)
            if inside:
                bmesh.ops.reverse_faces(bm, faces=faces)
        bm.normal_update()
        bm.to_mesh(object_.data)
        object_.data.update()
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to recalculate normals on '{object_.name}'."
        ) from exc
    finally:
        bm.free()


def remove_loose_geometry(object_: Object) -> int:
    """Removes loose vertices and edges not used by faces.

    Args:
        object_: Target mesh object.

    Returns:
        Number of removed elements.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("remove_loose_geometry requires a mesh object.")

    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        loose_verts = [vert for vert in bm.verts if not vert.link_faces and not vert.link_edges]
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        removed = len(loose_verts) + len(loose_edges)
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
        bm.to_mesh(object_.data)
        object_.data.update()
        return removed
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove loose geometry on '{object_.name}'."
        ) from exc
    finally:
        bm.free()


def triangulate_if_needed(object_: Object) -> bool:
    """Triangulates non-triangular faces in a mesh.

    Args:
        object_: Target mesh object.

    Returns:
        ``True`` when triangulation modified the mesh.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("triangulate_if_needed requires a mesh object.")

    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        faces_to_triangulate = [face for face in bm.faces if len(face.verts) > 3]
        if not faces_to_triangulate:
            return False
        bmesh.ops.triangulate(
            bm,
            faces=faces_to_triangulate,
            quad_method="BEAUTY",
            ngon_method="BEAUTY",
        )
        bm.to_mesh(object_.data)
        object_.data.update()
        return True
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to triangulate mesh '{object_.name}'."
        ) from exc
    finally:
        bm.free()


def get_principled_materials(object_: Optional[Object] = None) -> List[Material]:
    """Returns materials that use a Principled BSDF shader.

    Args:
        object_: Optional mesh object to inspect. When omitted, scan all materials.

    Returns:
        Matching material list.
    """
    _require_blender()
    materials: Sequence[Material]
    if object_ is None:
        materials = bpy.data.materials
    elif object_.type == "MESH":
        materials = [slot.material for slot in object_.material_slots if slot.material]
    else:
        raise TypeError("get_principled_materials expects a mesh object or None.")

    result: List[Material] = []
    for material in materials:
        if material is None or not material.use_nodes or material.node_tree is None:
            continue
        if any(node.type == "BSDF_PRINCIPLED" for node in material.node_tree.nodes):
            result.append(material)
    return result


def assign_default_material(object_: Object, material_name: Optional[str] = None) -> Material:
    """Assigns the configured default material to a mesh object.

    Args:
        object_: Target mesh object.
        material_name: Optional material override name.

    Returns:
        The assigned material.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("assign_default_material requires a mesh object.")

    defaults = DEFAULT_CONFIG.materials
    name = material_name or defaults.name
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = defaults.use_nodes
    material.use_backface_culling = defaults.double_sided is False

    if material.use_nodes and material.node_tree:
        principled = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            material.node_tree.nodes.clear()
            output = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
            principled = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
            material.node_tree.links.new(principled.outputs[0], output.inputs[0])
        for input_name, value in defaults.principled_inputs().items():
            if input_name in principled.inputs:
                principled.inputs[input_name].default_value = value
    material.diffuse_color = defaults.base_color

    if object_.data.materials:
        object_.data.materials[0] = material
    else:
        object_.data.materials.append(material)
    return material


def remove_unused_materials() -> int:
    """Removes orphaned materials from the Blender file.

    Returns:
        Number of removed materials.
    """
    _require_blender()
    unused = [material for material in bpy.data.materials if material.users == 0]
    for material in unused:
        bpy.data.materials.remove(material)
    return len(unused)


def has_uv(object_: Object) -> bool:
    """Checks whether a mesh object has UV layers."""
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("has_uv requires a mesh object.")
    return bool(object_.data.uv_layers)


def has_vertex_groups(object_: Object) -> bool:
    """Checks whether a mesh object has at least one vertex group."""
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("has_vertex_groups requires a mesh object.")
    return len(object_.vertex_groups) > 0


def count_triangles(object_: Object, evaluated: bool = True) -> int:
    """Counts mesh triangles.

    Args:
        object_: Target mesh object.
        evaluated: When ``True``, include modifiers via evaluated mesh.

    Returns:
        Triangle count.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("count_triangles requires a mesh object.")

    mesh, evaluated_object = _get_mesh_for_counting(object_, evaluated=evaluated)
    try:
        mesh.calc_loop_triangles()
        return len(mesh.loop_triangles)
    finally:
        _release_temp_mesh(object_, mesh, evaluated_object)


def count_vertices(object_: Object, evaluated: bool = True) -> int:
    """Counts mesh vertices.

    Args:
        object_: Target mesh object.
        evaluated: When ``True``, include modifiers via evaluated mesh.

    Returns:
        Vertex count.
    """
    _require_blender()
    if object_.type != "MESH":
        raise TypeError("count_vertices requires a mesh object.")

    mesh, evaluated_object = _get_mesh_for_counting(object_, evaluated=evaluated)
    try:
        return len(mesh.vertices)
    finally:
        _release_temp_mesh(object_, mesh, evaluated_object)


def _get_mesh_for_counting(object_: Object, evaluated: bool):
    """Returns mesh data for topology inspection."""
    if not evaluated:
        return object_.data, None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = object_.evaluated_get(depsgraph)
    return evaluated_object.to_mesh(), evaluated_object


def _release_temp_mesh(object_: Object, mesh, evaluated_object) -> None:
    """Releases temporary evaluated mesh data when needed."""
    if evaluated_object is not None:
        evaluated_object.to_mesh_clear()


def make_session_id() -> str:
    """Returns a compact unique identifier useful for reports or temp names."""
    return uuid.uuid4().hex[:12]
