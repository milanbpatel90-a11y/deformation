from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

ADDON_NAME = "Defirmation Template Factory"
ADDON_PACKAGE = "designer.blender_addon"
ADDON_VERSION: Tuple[int, int, int] = (0, 1, 0)
BLENDER_VERSION_MIN: Tuple[int, int, int] = (4, 0, 0)
DEFAULT_TEMPLATE_WIDTH_MM = 140.0
DEFAULT_EXPORT_FORMAT = "GLB"
DEFAULT_IMPORT_FORMAT = "GLB"

SUPPORTED_IMPORT_FORMATS: Tuple[str, ...] = (
    "GLB",
    "GLTF",
    "OBJ",
    "FBX",
    "STL",
    "PLY",
)
SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = (
    "GLB",
    "GLTF",
    "OBJ",
    "FBX",
)


@dataclass(frozen=True, slots=True)
class DirectoryConfig:
    """Stores relative working directories for the add-on.

    Attributes:
        imports: Directory name for imported source assets.
        cleaned: Directory name for cleaned intermediate assets.
        templates: Directory name for prepared template assets.
        exports: Directory name for exported deliverables.
        reports: Directory name for generated reports.
        textures: Directory name for textures and maps.
        logs: Directory name for log files.
    """

    imports: str = "imports"
    cleaned: str = "cleaned"
    templates: str = "templates"
    exports: str = "exports"
    reports: str = "reports"
    textures: str = "textures"
    logs: str = "logs"

    def as_dict(self) -> Dict[str, str]:
        """Returns the configured directory names as a dictionary."""
        return {
            "imports": self.imports,
            "cleaned": self.cleaned,
            "templates": self.templates,
            "exports": self.exports,
            "reports": self.reports,
            "textures": self.textures,
            "logs": self.logs,
        }

    def resolve(self, base_dir: str | Path) -> Dict[str, Path]:
        """Resolves configured directories relative to a base directory.

        Args:
            base_dir: Base workspace directory.

        Returns:
            Mapping of directory labels to absolute paths.
        """
        root = Path(base_dir).expanduser().resolve()
        return {name: root / value for name, value in self.as_dict().items()}


@dataclass(frozen=True, slots=True)
class FactoryDirectoryConfig:
    """Offline template-factory directory layout."""

    raw: str = "templates/raw"
    processed: str = "templates/processed"
    descriptors: str = "templates/descriptors"
    metadata: str = "templates/metadata"
    thumbnails: str = "templates/thumbnails"
    registry: str = "templates/registry.json"

    def as_dict(self) -> Dict[str, str]:
        return {
            "raw": self.raw,
            "processed": self.processed,
            "descriptors": self.descriptors,
            "metadata": self.metadata,
            "thumbnails": self.thumbnails,
            "registry": self.registry,
        }

    def resolve(self, base_dir: str | Path) -> Dict[str, Path]:
        root = Path(base_dir).expanduser().resolve()
        return {name: root / value for name, value in self.as_dict().items()}


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """Validation tolerances used by geometry and export checks."""

    min_width_mm: float = 110.0
    max_width_mm: float = 170.0
    default_target_width_mm: float = DEFAULT_TEMPLATE_WIDTH_MM
    min_vertex_count: int = 100
    max_vertex_count: int = 250_000
    max_triangle_count: int = 500_000
    merge_distance_mm: float = 0.01
    non_manifold_ratio_limit: float = 0.02
    degenerate_face_area_mm2: float = 0.0001
    bounding_box_epsilon_mm: float = 0.1
    require_uvs: bool = True
    require_vertex_groups: bool = True


@dataclass(frozen=True, slots=True)
class MaterialDefaults:
    """Default physically based material settings for generated templates."""

    name: str = "TemplateMaterial"
    base_color: Tuple[float, float, float, float] = (0.08, 0.08, 0.08, 1.0)
    metallic: float = 0.0
    roughness: float = 0.45
    specular_ior_level: float = 0.5
    transmission_weight: float = 0.0
    alpha: float = 1.0
    use_nodes: bool = True
    double_sided: bool = False

    def principled_inputs(self) -> Dict[str, float | Tuple[float, float, float, float]]:
        """Returns defaults keyed by Principled BSDF input name."""
        return {
            "Base Color": self.base_color,
            "Metallic": self.metallic,
            "Roughness": self.roughness,
            "Specular IOR Level": self.specular_ior_level,
            "Transmission Weight": self.transmission_weight,
            "Alpha": self.alpha,
        }


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging settings shared across add-on modules."""

    logger_name: str = "defirmation.template_factory"
    level: str = "INFO"
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    propagate: bool = False
    capture_warnings: bool = True
    file_name: str = "template_factory.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Default export behavior for template delivery."""

    default_format: str = DEFAULT_EXPORT_FORMAT
    embed_textures: bool = True
    apply_modifiers: bool = True
    export_y_up: bool = True
    use_selection_only: bool = False
    compress: bool = False


@dataclass(frozen=True, slots=True)
class ImportConfig:
    """Default import behavior for source CAD and mesh assets."""

    default_format: str = DEFAULT_IMPORT_FORMAT
    merge_vertices_on_import: bool = False
    validate_meshes: bool = True
    auto_unit_scale: bool = True


@dataclass(frozen=True, slots=True)
class AddonMetadata:
    """Metadata describing the Blender add-on package."""

    name: str = ADDON_NAME
    package: str = ADDON_PACKAGE
    version: Tuple[int, int, int] = ADDON_VERSION
    blender_min_version: Tuple[int, int, int] = BLENDER_VERSION_MIN
    author: str = "Defirmation"
    category: str = "Import-Export"
    description: str = (
        "Prepare, validate, standardize, and export eyewear template models."
    )

    @property
    def version_string(self) -> str:
        """Returns the semantic version string."""
        return ".".join(str(part) for part in self.version)


@dataclass(frozen=True, slots=True)
class AddonConfig:
    """Top-level immutable configuration container for the add-on."""

    metadata: AddonMetadata = field(default_factory=AddonMetadata)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    factory: FactoryDirectoryConfig = field(default_factory=FactoryDirectoryConfig)
    validation: ValidationThresholds = field(default_factory=ValidationThresholds)
    materials: MaterialDefaults = field(default_factory=MaterialDefaults)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    importer: ImportConfig = field(default_factory=ImportConfig)
    supported_import_formats: Sequence[str] = SUPPORTED_IMPORT_FORMATS
    supported_export_formats: Sequence[str] = SUPPORTED_EXPORT_FORMATS

    def resolve_directories(
        self,
        base_dir: str | Path,
        create: bool = False,
    ) -> Dict[str, Path]:
        """Resolves working directories and optionally creates them.

        Args:
            base_dir: Base workspace directory.
            create: When ``True``, create missing directories.

        Returns:
            Mapping of directory labels to absolute paths.
        """
        resolved = self.directories.resolve(base_dir)
        if create:
            for path in resolved.values():
                path.mkdir(parents=True, exist_ok=True)
        return resolved

    def resolve_factory_directories(
        self,
        base_dir: str | Path,
        create: bool = False,
    ) -> Dict[str, Path]:
        """Resolves template-factory directories and optionally creates them."""
        resolved = self.factory.resolve(base_dir)
        if create:
            for name, path in resolved.items():
                if name == "registry":
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    path.mkdir(parents=True, exist_ok=True)
        return resolved


DEFAULT_CONFIG = AddonConfig()


def get_default_config() -> AddonConfig:
    """Returns the immutable default add-on configuration."""
    return DEFAULT_CONFIG


def resolve_workspace_root(base_dir: Optional[str | Path] = None) -> Path:
    """Resolves the workspace root used by the add-on.

    Args:
        base_dir: Optional explicit workspace root.

    Returns:
        Absolute workspace path.
    """
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    return Path.cwd().resolve()
