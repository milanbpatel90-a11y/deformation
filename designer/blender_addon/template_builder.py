from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Iterable, Optional

from .config import DEFAULT_CONFIG
from .descriptor_generator import generate_descriptor
from .feature_detector import TemplateAnalyzer
from .importer import import_asset
from .material_analyzer import convert_to_pbr
from .materials import fix_material_names, merge_duplicate_materials, remove_unused_materials
from .metadata_generator import generate_metadata
from .normalize import normalize_objects
from .quality_validator import validate_template
from .registry_builder import TemplateRegistryBuilder
from .repair import repair_objects
from .thumbnail_renderer import render_thumbnails
from .utils import ensure_directory, get_logger, list_glb_files, safe_export_path
from .uv_validator import repair_uv
from .vertex_groups import build_vertex_groups

try:
    import bpy
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TemplateBuildReport:
    """Summarizes template build results."""

    source_file: str
    processing_time: float
    repair_statistics: dict[str, object] = field(default_factory=dict)
    validation_score: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)


def _reset_scene() -> None:
    """Clears the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _import_glb(path: Path) -> list[Object]:
    """Imports a GLB file and returns imported mesh objects."""
    return import_asset(path).imported_meshes


def _export_glb(path: Path, objects: Iterable[Object]) -> Path:
    """Exports the given mesh objects to GLB."""
    bpy.ops.object.select_all(action="DESELECT")
    for object_ in objects:
        object_.select_set(True)
    bpy.context.view_layer.objects.active = next(iter(objects), None)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    return path


def _normalize_objects(objects: Iterable[Object]) -> None:
    """Applies transform normalization to imported meshes."""
    normalize_objects(objects, DEFAULT_CONFIG.validation.default_target_width_mm)


def _standardize_materials(objects: Iterable[Object]) -> None:
    """Normalizes materials for GLB export."""
    for object_ in objects:
        for slot in object_.material_slots:
            if slot.material is not None:
                convert_to_pbr(slot.material)
    fix_material_names()
    merge_duplicate_materials(objects)
    remove_unused_materials()


def build_template(
    source_file: str | Path,
    output_dir: Optional[str | Path] = None,
    template_id: Optional[str] = None,
) -> TemplateBuildReport:
    """Builds a single standardized template from a source GLB."""
    start_time = time.perf_counter()
    source_path = Path(source_file).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else source_path.parent
    ensure_directory(target_dir)
    template_name = template_id or source_path.stem

    _reset_scene()
    mesh_objects = _import_glb(source_path)
    repair_statistics: dict[str, object] = {}
    warnings: list[str] = []
    errors: list[str] = []

    if not mesh_objects:
        return TemplateBuildReport(
            source_file=str(source_path),
            processing_time=time.perf_counter() - start_time,
            repair_statistics={},
            validation_score=0,
            warnings=[],
            errors=["No mesh objects imported from source file."],
            output_files=[],
        )

    repair_batch = repair_objects(mesh_objects)
    for object_name, report in repair_batch.reports.items():
        repair_statistics[object_name] = asdict(report)

    _normalize_objects(mesh_objects)
    initial_validation = validate_template(mesh_objects)
    warnings.extend(initial_validation.warnings)
    errors.extend(initial_validation.errors)

    analyzer = TemplateAnalyzer(mesh_objects)
    analysis = analyzer.analyze()
    vertex_groups = build_vertex_groups(mesh_objects)
    uv_reports = repair_uv(mesh_objects)
    _standardize_materials(mesh_objects)
    final_validation = validate_template(mesh_objects)

    has_vertex_groups = bool(vertex_groups.groups_created)
    has_uv = all(report.valid for report in uv_reports.values()) if uv_reports else False
    descriptor = generate_descriptor(mesh_objects)
    metadata_artifacts = generate_metadata(
        template_id=template_name,
        analysis=analysis,
        validation=final_validation,
        descriptor=descriptor,
        has_vertex_groups=has_vertex_groups,
        has_uv=has_uv,
    )

    export_path = safe_export_path(target_dir, template_name, ".glb", overwrite=True)
    metadata_path = export_path.with_suffix(".metadata.json")
    descriptor_path = export_path.with_suffix(".descriptor.json")
    _export_glb(export_path, mesh_objects)
    metadata_artifacts.metadata.write_json(metadata_path)
    descriptor.write_json(descriptor_path)

    registry_builder = TemplateRegistryBuilder(target_dir)
    registry_path = registry_builder.build_registry()
    thumbnails = render_thumbnails(template_name, target_dir)

    output_files = [
        str(export_path),
        str(metadata_path),
        str(descriptor_path),
        str(registry_path),
        str(thumbnails.front),
        str(thumbnails.side),
        str(thumbnails.perspective),
    ]
    return TemplateBuildReport(
        source_file=str(source_path),
        processing_time=time.perf_counter() - start_time,
        repair_statistics={
            "mesh_repairs": repair_statistics,
            "vertex_groups": vertex_groups.vertex_counts,
            "uv_reports": {
                name: {
                    "valid": report.valid,
                    "missing_faces": report.missing_faces,
                    "overlapping_faces": report.overlapping_faces,
                    "flipped_faces": report.flipped_faces,
                }
                for name, report in uv_reports.items()
            },
        },
        validation_score=final_validation.score,
        warnings=sorted(set(warnings + final_validation.warnings)),
        errors=sorted(set(errors + final_validation.errors)),
        output_files=output_files,
    )


def ensure_factory_structure(workspace_root: str | Path) -> dict[str, Path]:
    """Creates the offline template-factory directory layout."""
    factory_paths = DEFAULT_CONFIG.resolve_factory_directories(workspace_root, create=True)
    registry_path = factory_paths["registry"]
    if not registry_path.exists():
        registry_path.write_text("[]\n", encoding="utf-8")
    return factory_paths


def build_template_from_file(
    source_file: str | Path,
    workspace_root: str | Path,
    template_id: Optional[str] = None,
) -> TemplateBuildReport:
    """Runs the factory flow using the configured processed output folder."""
    factory_paths = ensure_factory_structure(workspace_root)
    report = build_template(
        source_file=source_file,
        output_dir=factory_paths["processed"],
        template_id=template_id,
    )

    source_path = Path(source_file).expanduser().resolve()
    template_name = template_id or source_path.stem
    processed_glb = factory_paths["processed"] / f"{template_name}.glb"
    processed_metadata = processed_glb.with_suffix(".metadata.json")
    processed_descriptor = processed_glb.with_suffix(".descriptor.json")

    if processed_descriptor.exists():
        descriptor_target = factory_paths["descriptors"] / f"{template_name}.json"
        descriptor_target.write_text(processed_descriptor.read_text(encoding="utf-8"), encoding="utf-8")
    if processed_metadata.exists():
        metadata_target = factory_paths["metadata"] / f"{template_name}.json"
        metadata_target.write_text(processed_metadata.read_text(encoding="utf-8"), encoding="utf-8")

    registry_builder = TemplateRegistryBuilder(factory_paths["processed"])
    entries = registry_builder.scan_templates()
    registry_payload = []
    for entry in entries:
        item = asdict(entry)
        descriptor_rel = f"descriptors/{Path(entry.id).stem}.json"
        metadata_rel = f"metadata/{Path(entry.id).stem}.json"
        item["descriptor"] = descriptor_rel
        item["metadata_path"] = metadata_rel
        registry_payload.append(item)
    factory_paths["registry"].write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_folder(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> list[TemplateBuildReport]:
    """Builds templates for every GLB file in a folder."""
    input_path = Path(input_dir).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else input_path
    ensure_directory(target_dir)
    reports: list[TemplateBuildReport] = []
    for source_file in list_glb_files(input_path, recursive=True):
        reports.append(build_template(source_file, target_dir))
    return reports


def batch_process(
    files: Iterable[str | Path],
    output_dir: str | Path,
) -> list[TemplateBuildReport]:
    """Batch-processes an arbitrary iterable of source GLB files."""
    target_dir = Path(output_dir).expanduser().resolve()
    ensure_directory(target_dir)
    return [build_template(file_path, target_dir) for file_path in files]


__all__ = [
    "TemplateBuildReport",
    "batch_process",
    "build_folder",
    "build_template",
    "build_template_from_file",
    "ensure_factory_structure",
]
