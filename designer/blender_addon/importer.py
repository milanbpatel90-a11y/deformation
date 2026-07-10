from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG
from .utils import _require_blender, get_logger

try:
    import bpy
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ImportReport:
    """Result of importing a source asset into Blender."""

    source_file: str
    format: str
    imported_meshes: list[Object] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.imported_meshes)


def verify_source_file(source_file: str | Path) -> Path:
    """Validates the incoming asset path before import."""
    source_path = Path(source_file).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")

    suffix = source_path.suffix.lstrip(".").upper()
    if suffix not in DEFAULT_CONFIG.supported_import_formats:
        raise ValueError(
            f"Unsupported import format '{suffix}'. "
            f"Expected one of {DEFAULT_CONFIG.supported_import_formats}."
        )
    return source_path


def import_asset(source_file: str | Path) -> ImportReport:
    """Imports a supported source file and returns imported mesh objects."""
    _require_blender()
    source_path = verify_source_file(source_file)
    before = {object_.name for object_ in bpy.data.objects}
    suffix = source_path.suffix.lower()

    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(source_path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source_path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source_path))
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(source_path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(source_path))
    else:  # pragma: no cover - protected by verify_source_file
        raise ValueError(f"Unsupported import format: {suffix}")

    imported_meshes = [
        object_
        for object_ in bpy.data.objects
        if object_.name not in before and object_.type == "MESH"
    ]
    warnings: list[str] = []
    if imported_meshes and not any(object_.material_slots for object_ in imported_meshes):
        warnings.append("Imported mesh has no material slots.")

    return ImportReport(
        source_file=str(source_path),
        format=source_path.suffix.lstrip(".").upper(),
        imported_meshes=imported_meshes,
        warnings=warnings,
    )

