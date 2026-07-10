from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence

from .config import DEFAULT_CONFIG
from .utils import (
    _require_blender,
    assign_default_material as _assign_default_material,
    generate_unique_name,
    get_logger,
    get_principled_materials as _get_principled_materials,
    remove_unused_materials as _remove_unused_materials,
)

try:
    import bpy
    from bpy.types import Image, Material, Object
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Image = object  # type: ignore[assignment]
    Material = object  # type: ignore[assignment]
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MaterialValidationReport:
    """Summarizes material validation results."""

    is_valid: bool
    material_count: int
    principled_count: int
    duplicate_count: int
    missing_texture_count: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def _mesh_objects(objects: Optional[Iterable[Object]] = None) -> list[Object]:
    """Returns mesh objects from the provided iterable or active scene."""
    _require_blender()
    source = list(objects) if objects is not None else list(bpy.context.scene.objects)
    return [object_ for object_ in source if object_.type == "MESH"]


def _material_slots(objects: Iterable[Object]) -> list[tuple[Object, int, Material]]:
    """Returns mesh material slots with materials attached."""
    slots: list[tuple[Object, int, Material]] = []
    for object_ in objects:
        for index, slot in enumerate(object_.material_slots):
            if slot.material is not None:
                slots.append((object_, index, slot.material))
    return slots


def _material_output(material: Material):
    """Returns or creates the material output node."""
    if material.node_tree is None:
        material.use_nodes = True
    node_tree = material.node_tree
    output = next(
        (node for node in node_tree.nodes if node.type == "OUTPUT_MATERIAL"),
        None,
    )
    if output is None:
        output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    return output


def _principled_node(material: Material):
    """Returns or creates the Principled BSDF node."""
    material.use_nodes = True
    node_tree = material.node_tree
    principled = next(
        (node for node in node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is None:
        principled = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.location = (0.0, 0.0)
    return principled


def _copy_socket_default(source_socket, target_socket) -> None:
    """Copies a default socket value when compatible."""
    try:
        target_socket.default_value = source_socket.default_value
    except Exception:
        return


def _material_signature(material: Material) -> tuple:
    """Builds a hashable signature for material deduplication."""
    principled = None
    textures: list[tuple[str, str]] = []
    if material.use_nodes and material.node_tree is not None:
        principled = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or getattr(node, "image", None) is None:
                continue
            image = node.image
            textures.append((node.label or node.name, Path(image.filepath_from_user()).name))

    principled_values = ()
    if principled is not None:
        inputs = principled.inputs
        principled_values = (
            tuple(inputs["Base Color"].default_value),
            float(inputs["Metallic"].default_value),
            float(inputs["Roughness"].default_value),
            float(inputs["Transmission Weight"].default_value)
            if "Transmission Weight" in inputs
            else 0.0,
            float(inputs["Alpha"].default_value),
        )

    return (
        tuple(round(value, 6) for value in material.diffuse_color),
        principled_values,
        tuple(sorted(textures)),
        bool(material.use_backface_culling),
        int(material.blend_method != "OPAQUE"),
    )


def _sanitize_material_name(name: str) -> str:
    """Returns a Blender-safe material name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._") or "Material"


def get_principled_materials(object_: Optional[Object] = None) -> list[Material]:
    """Returns materials that use a Principled BSDF shader."""
    return _get_principled_materials(object_)


def convert_to_principled(material: Material) -> Material:
    """Converts a material node tree to a Principled BSDF workflow."""
    _require_blender()
    material.use_nodes = True
    node_tree = material.node_tree
    output = _material_output(material)
    principled = _principled_node(material)

    if not output.inputs["Surface"].is_linked:
        node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        return material

    linked_socket = output.inputs["Surface"].links[0].from_socket
    source_node = linked_socket.node
    if source_node.type == "BSDF_PRINCIPLED":
        return material

    if "Color" in getattr(source_node, "inputs", {}):
        _copy_socket_default(source_node.inputs["Color"], principled.inputs["Base Color"])
    elif "Base Color" in getattr(source_node, "inputs", {}):
        _copy_socket_default(
            source_node.inputs["Base Color"],
            principled.inputs["Base Color"],
        )
    if "Roughness" in getattr(source_node, "inputs", {}):
        _copy_socket_default(
            source_node.inputs["Roughness"],
            principled.inputs["Roughness"],
        )
    if "Metallic" in getattr(source_node, "inputs", {}):
        _copy_socket_default(
            source_node.inputs["Metallic"],
            principled.inputs["Metallic"],
        )
    if "Alpha" in getattr(source_node, "inputs", {}):
        _copy_socket_default(source_node.inputs["Alpha"], principled.inputs["Alpha"])

    for input_socket in source_node.inputs:
        if not input_socket.is_linked:
            continue
        link = input_socket.links[0]
        if input_socket.name in {"Color", "Base Color"}:
            node_tree.links.new(link.from_socket, principled.inputs["Base Color"])
        elif input_socket.name == "Roughness":
            node_tree.links.new(link.from_socket, principled.inputs["Roughness"])
        elif input_socket.name == "Metallic":
            node_tree.links.new(link.from_socket, principled.inputs["Metallic"])
        elif input_socket.name == "Normal" and "Normal" in principled.inputs:
            node_tree.links.new(link.from_socket, principled.inputs["Normal"])
        elif input_socket.name == "Alpha" and "Alpha" in principled.inputs:
            node_tree.links.new(link.from_socket, principled.inputs["Alpha"])

    node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def assign_default_material(
    object_: Object,
    material_name: Optional[str] = None,
) -> Material:
    """Assigns the configured default material to a mesh object."""
    return _assign_default_material(object_, material_name=material_name)


def merge_duplicate_materials(objects: Optional[Iterable[Object]] = None) -> int:
    """Merges geometrically identical materials across mesh objects."""
    _require_blender()
    mesh_objects = _mesh_objects(objects)
    slots = _material_slots(mesh_objects)
    canonical_by_signature: dict[tuple, Material] = {}
    duplicate_replacements = 0

    for object_, slot_index, material in slots:
        signature = _material_signature(material)
        canonical = canonical_by_signature.get(signature)
        if canonical is None:
            canonical_by_signature[signature] = material
            continue
        if canonical == material:
            continue
        object_.material_slots[slot_index].material = canonical
        duplicate_replacements += 1

    _remove_unused_materials()
    return duplicate_replacements


def remove_unused_materials() -> int:
    """Removes orphaned materials from the current Blender file."""
    return _remove_unused_materials()


def fix_material_names(materials: Optional[Iterable[Material]] = None) -> int:
    """Normalizes material names and ensures uniqueness."""
    _require_blender()
    source = list(materials) if materials is not None else list(bpy.data.materials)
    used_names: set[str] = set()
    renamed = 0
    for material in source:
        sanitized = _sanitize_material_name(material.name)
        unique = generate_unique_name(sanitized, used_names)
        used_names.add(unique)
        if material.name != unique:
            material.name = unique
            renamed += 1
    return renamed


def validate_materials(objects: Optional[Iterable[Object]] = None) -> MaterialValidationReport:
    """Validates material compatibility for GLB export."""
    _require_blender()
    mesh_objects = _mesh_objects(objects)
    materials = {
        slot.material
        for object_ in mesh_objects
        for slot in object_.material_slots
        if slot.material is not None
    }
    principled_materials = {material for material in materials if material in _get_principled_materials()}
    warnings: list[str] = []
    errors: list[str] = []

    if not materials:
        warnings.append("No materials assigned to mesh objects.")

    signature_counts: dict[tuple, int] = {}
    missing_texture_count = 0
    for material in materials:
        signature = _material_signature(material)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        if material not in principled_materials:
            errors.append(f"Material '{material.name}' is not Principled BSDF based.")
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or getattr(node, "image", None) is None:
                continue
            image = node.image
            if image.packed_file is not None:
                continue
            image_path = Path(image.filepath_from_user())
            if image_path.exists():
                continue
            missing_texture_count += 1
            warnings.append(
                f"Material '{material.name}' references missing texture '{image_path.name}'."
            )

    duplicate_count = sum(count - 1 for count in signature_counts.values() if count > 1)
    if duplicate_count:
        warnings.append(f"Detected {duplicate_count} duplicate material instances.")

    return MaterialValidationReport(
        is_valid=not errors,
        material_count=len(materials),
        principled_count=len(principled_materials),
        duplicate_count=duplicate_count,
        missing_texture_count=missing_texture_count,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def convert_to_pbr(material: Material) -> Material:
    """Converts a material to a GLB-friendly PBR configuration."""
    _require_blender()
    material = convert_to_principled(material)
    defaults = DEFAULT_CONFIG.materials
    material.use_backface_culling = defaults.double_sided is False
    material.blend_method = "OPAQUE" if defaults.alpha >= 1.0 else "BLEND"
    material.shadow_method = "OPAQUE" if defaults.alpha >= 1.0 else "CLIP"

    principled = _principled_node(material)
    for input_name, value in defaults.principled_inputs().items():
        if input_name not in principled.inputs:
            continue
        socket = principled.inputs[input_name]
        if socket.is_linked:
            continue
        socket.default_value = value
    material.diffuse_color = defaults.base_color
    return material


def generate_default_pbr(name: Optional[str] = None) -> Material:
    """Creates a default Principled PBR material."""
    _require_blender()
    material_name = name or DEFAULT_CONFIG.materials.name
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
    return convert_to_pbr(material)


def copy_materials(
    source_object: Object,
    target_objects: Iterable[Object],
    duplicate_materials: bool = True,
) -> int:
    """Copies material assignments from one object to one or more target objects."""
    _require_blender()
    if source_object.type != "MESH":
        raise TypeError("Source object must be a mesh object.")

    source_materials = [slot.material for slot in source_object.material_slots if slot.material]
    copied_assignments = 0
    for target_object in target_objects:
        if target_object.type != "MESH":
            continue
        target_object.data.materials.clear()
        for material in source_materials:
            assigned = material.copy() if duplicate_materials else material
            target_object.data.materials.append(assigned)
            copied_assignments += 1
    return copied_assignments


def repair_missing_textures(
    search_directories: Optional[Iterable[str | Path]] = None,
    objects: Optional[Iterable[Object]] = None,
) -> int:
    """Repairs broken image texture filepaths by searching known directories."""
    _require_blender()
    directories = [
        Path(directory).expanduser().resolve()
        for directory in (search_directories or [])
    ]
    repaired = 0

    mesh_objects = _mesh_objects(objects)
    materials = {
        slot.material
        for object_ in mesh_objects
        for slot in object_.material_slots
        if slot.material is not None
    }
    for material in materials:
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or getattr(node, "image", None) is None:
                continue
            image: Image = node.image
            if image.packed_file is not None:
                continue
            current_path = Path(image.filepath_from_user())
            if current_path.exists():
                continue
            for directory in directories:
                candidate = directory / current_path.name
                if candidate.exists():
                    image.filepath = str(candidate)
                    repaired += 1
                    break
    return repaired


__all__ = [
    "MaterialValidationReport",
    "assign_default_material",
    "convert_to_pbr",
    "convert_to_principled",
    "copy_materials",
    "fix_material_names",
    "generate_default_pbr",
    "get_principled_materials",
    "merge_duplicate_materials",
    "remove_unused_materials",
    "repair_missing_textures",
    "validate_materials",
]
