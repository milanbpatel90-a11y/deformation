from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .geometry import BoundingBox, calculate_bounding_box
from .utils import _require_blender, get_logger

try:
    import bmesh
    from bpy.types import Mesh, Object
    from mathutils import Vector
except ImportError:  # pragma: no cover - Blender runtime only
    bmesh = None
    Mesh = object  # type: ignore[assignment]
    Object = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)

FRAME = "Frame"
LEFT_LENS = "LeftLens"
RIGHT_LENS = "RightLens"
BRIDGE = "Bridge"
LEFT_TEMPLE = "LeftTemple"
RIGHT_TEMPLE = "RightTemple"
LEFT_PAD = "LeftPad"
RIGHT_PAD = "RightPad"

PART_NAMES = (
    FRAME,
    LEFT_LENS,
    RIGHT_LENS,
    BRIDGE,
    LEFT_TEMPLE,
    RIGHT_TEMPLE,
    LEFT_PAD,
    RIGHT_PAD,
)


@dataclass(slots=True)
class MeshPart:
    """Represents a logical eyewear mesh part."""

    name: str
    objects: list[Object] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, float | str | bool] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        """Returns the number of objects assigned to the part."""
        return len(self.objects)


def _mesh_objects(objects: Optional[Iterable[Object]] = None) -> list[Object]:
    """Returns mesh objects from input or the active scene."""
    _require_blender()
    source = list(objects) if objects is not None else list(__import__("bpy").context.scene.objects)
    return [object_ for object_ in source if object_.type == "MESH"]


def _world_bbox(object_: Object) -> BoundingBox:
    """Returns a world-space bounding box for an object."""
    return calculate_bounding_box(object_, world_space=True)


def _dimensions(bbox: BoundingBox) -> Vector:
    """Returns bbox size vector."""
    return bbox.size.copy()


def _name_tokens(name: str) -> set[str]:
    """Returns normalized object name tokens."""
    import re

    return {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", name.lower())
        if token
    }


def _component_meshes(object_: Object) -> list[Mesh]:
    """Splits disconnected mesh components into temporary Mesh datablocks."""
    bm = bmesh.new()
    try:
        bm.from_mesh(object_.data)
        bm.verts.ensure_lookup_table()

        visited: set[int] = set()
        meshes: list[Mesh] = []
        bpy_module = __import__("bpy")
        for vert in bm.verts:
            if vert.index in visited:
                continue
            stack = [vert]
            component_verts = set()
            component_edges = set()
            component_faces = set()
            while stack:
                current = stack.pop()
                if current.index in visited:
                    continue
                visited.add(current.index)
                component_verts.add(current)
                for edge in current.link_edges:
                    component_edges.add(edge)
                    for linked_vert in edge.verts:
                        if linked_vert.index not in visited:
                            stack.append(linked_vert)
                    for face in edge.link_faces:
                        component_faces.add(face)
                        for face_vert in face.verts:
                            if face_vert.index not in visited:
                                stack.append(face_vert)

            new_bm = bmesh.new()
            vert_map = {old_vert: new_bm.verts.new(old_vert.co.copy()) for old_vert in component_verts}
            new_bm.verts.ensure_lookup_table()
            for face in component_faces:
                try:
                    new_bm.faces.new([vert_map[vert] for vert in face.verts])
                except ValueError:
                    continue
            for edge in component_edges:
                verts = [vert_map[vert] for vert in edge.verts if vert in vert_map]
                if len(verts) == 2:
                    try:
                        new_bm.edges.new(verts)
                    except ValueError:
                        continue
            mesh = bpy_module.data.meshes.new(f"{object_.name}_component")
            new_bm.to_mesh(mesh)
            mesh.update()
            new_bm.free()
            meshes.append(mesh)
        return meshes
    finally:
        bm.free()


def _instantiate_component_object(source: Object, mesh: Mesh, index: int) -> Object:
    """Creates a temporary object for a split component."""
    bpy_module = __import__("bpy")
    new_object = bpy_module.data.objects.new(f"{source.name}_part_{index:03d}", mesh)
    new_object.matrix_world = source.matrix_world.copy()
    return new_object


def _has_lens_material(object_: Object) -> bool:
    """Heuristic for detecting lens-like materials."""
    for slot in object_.material_slots:
        material = slot.material
        if material is None:
            continue
        name = material.name.lower()
        if any(token in name for token in ("lens", "glass", "transparent", "clear")):
            return True
        if material.use_nodes and material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                transmission = (
                    node.inputs["Transmission Weight"].default_value
                    if "Transmission Weight" in node.inputs
                    else 0.0
                )
                alpha = node.inputs["Alpha"].default_value if "Alpha" in node.inputs else 1.0
                if float(transmission) > 0.2 or float(alpha) < 0.98:
                    return True
    return False


def _classify_object(object_: Object, global_bbox: BoundingBox) -> tuple[str, float, dict[str, float | str | bool]]:
    """Assigns an object to the most likely eyewear part."""
    bbox = _world_bbox(object_)
    size = _dimensions(bbox)
    center = bbox.center
    tokens = _name_tokens(object_.name)
    global_size = global_bbox.size
    x_ratio = 0.0 if global_size.x == 0.0 else abs(center.x - global_bbox.center.x) / max(global_size.x * 0.5, 1e-6)
    y_ratio = 0.0 if global_size.y == 0.0 else (center.y - global_bbox.min_corner.y) / max(global_size.y, 1e-6)
    z_ratio = 0.0 if global_size.z == 0.0 else abs(center.z - global_bbox.center.z) / max(global_size.z * 0.5, 1e-6)

    if {"lens", "glass"} & tokens or _has_lens_material(object_):
        part = LEFT_LENS if center.x < global_bbox.center.x else RIGHT_LENS
        return part, 0.92, {"x_ratio": x_ratio, "is_lens_material": True}

    if {"temple", "arm", "leg"} & tokens or y_ratio > 0.62:
        part = LEFT_TEMPLE if center.x < global_bbox.center.x else RIGHT_TEMPLE
        return part, 0.85, {"x_ratio": x_ratio, "rear_ratio": y_ratio}

    if {"pad", "nosepad", "nose", "support"} & tokens or (size.x < global_size.x * 0.12 and z_ratio < 0.2 and abs(center.x - global_bbox.center.x) < global_size.x * 0.3):
        part = LEFT_PAD if center.x < global_bbox.center.x else RIGHT_PAD
        return part, 0.8, {"x_ratio": x_ratio, "center_offset": abs(center.x - global_bbox.center.x)}

    if {"bridge"} & tokens or (size.x < global_size.x * 0.25 and abs(center.x - global_bbox.center.x) < global_size.x * 0.12):
        return BRIDGE, 0.75, {"width_ratio": 0.0 if global_size.x == 0.0 else size.x / global_size.x}

    if {"frame", "rim", "front"} & tokens:
        return FRAME, 0.82, {"front_ratio": y_ratio}

    return FRAME, 0.55, {"front_ratio": y_ratio}


def split_eyewear_mesh(
    objects: Optional[Iterable[Object]] = None,
    split_components: bool = True,
) -> dict[str, MeshPart]:
    """Splits or classifies eyewear meshes into logical parts."""
    mesh_objects = _mesh_objects(objects)
    if not mesh_objects:
        return {name: MeshPart(name=name) for name in PART_NAMES}

    working_objects: list[Object] = []
    temporary_meshes: list[Mesh] = []
    if split_components:
        for object_ in mesh_objects:
            component_meshes = _component_meshes(object_)
            if len(component_meshes) <= 1:
                for mesh in component_meshes:
                    temporary_meshes.append(mesh)
                working_objects.append(object_)
                continue
            for index, mesh in enumerate(component_meshes):
                temporary_meshes.append(mesh)
                working_objects.append(_instantiate_component_object(object_, mesh, index))
    else:
        working_objects = mesh_objects

    global_bbox = calculate_bounding_box(working_objects[0], world_space=True)
    for object_ in working_objects[1:]:
        bbox = calculate_bounding_box(object_, world_space=True)
        min_corner = Vector((
            min(global_bbox.min_corner.x, bbox.min_corner.x),
            min(global_bbox.min_corner.y, bbox.min_corner.y),
            min(global_bbox.min_corner.z, bbox.min_corner.z),
        ))
        max_corner = Vector((
            max(global_bbox.max_corner.x, bbox.max_corner.x),
            max(global_bbox.max_corner.y, bbox.max_corner.y),
            max(global_bbox.max_corner.z, bbox.max_corner.z),
        ))
        size = max_corner - min_corner
        center = (min_corner + max_corner) * 0.5
        global_bbox = BoundingBox(min_corner=min_corner, max_corner=max_corner, center=center, size=size)

    parts = {name: MeshPart(name=name) for name in PART_NAMES}
    for object_ in working_objects:
        part_name, confidence, metadata = _classify_object(object_, global_bbox)
        part = parts[part_name]
        part.objects.append(object_)
        part.confidence = max(part.confidence, confidence)
        part.metadata.update(metadata)

    return parts


__all__ = [
    "BRIDGE",
    "FRAME",
    "LEFT_LENS",
    "LEFT_PAD",
    "LEFT_TEMPLE",
    "MeshPart",
    "PART_NAMES",
    "RIGHT_LENS",
    "RIGHT_PAD",
    "RIGHT_TEMPLE",
    "split_eyewear_mesh",
]
