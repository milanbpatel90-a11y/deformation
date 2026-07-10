from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .template_metadata import TemplateMetadata
from .utils import ensure_directory, get_logger, list_glb_files


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RegistryEntry:
    """Represents one template entry in the registry."""

    id: str
    template_path: str
    thumbnail: str
    preview: str
    metadata: dict[str, Any]
    quality: int
    style: str
    material: str
    frame_family: str
    bridge: str
    rim: str
    triangle_count: int
    vertex_count: int
    validation_score: int


class TemplateRegistryBuilder:
    """Scans template output folders and builds searchable registries."""

    def __init__(self, templates_dir: str | Path) -> None:
        """Initializes the registry builder."""
        self.templates_dir = Path(templates_dir).expanduser().resolve()
        self.registry_path = self.templates_dir / "registry.json"
        self.entries: list[RegistryEntry] = []

    def scan_templates(self) -> list[RegistryEntry]:
        """Scans the template directory and builds registry entries."""
        entries: list[RegistryEntry] = []
        for glb_path in list_glb_files(self.templates_dir, recursive=True):
            metadata_path = glb_path.with_suffix(".metadata.json")
            if not metadata_path.exists():
                metadata_path = glb_path.with_suffix(".json")
            if not metadata_path.exists():
                continue
            metadata = TemplateMetadata.read_json(metadata_path)
            thumbnail = glb_path.with_suffix(".thumbnail.png")
            preview = glb_path.with_suffix(".preview.png")
            entry = RegistryEntry(
                id=metadata.id,
                template_path=str(glb_path),
                thumbnail=str(thumbnail) if thumbnail.exists() else "",
                preview=str(preview) if preview.exists() else "",
                metadata=metadata.to_dict(),
                quality=metadata.quality_score,
                style=metadata.style,
                material=metadata.material,
                frame_family=metadata.family,
                bridge=metadata.bridge,
                rim=metadata.rim,
                triangle_count=metadata.triangle_count,
                vertex_count=metadata.vertex_count,
                validation_score=metadata.validation_score or metadata.quality_score,
            )
            entries.append(entry)
        self.entries = sorted(entries, key=lambda item: (-item.quality, item.id))
        return self.entries

    def build_registry(self) -> Path:
        """Builds and writes the registry file."""
        ensure_directory(self.templates_dir)
        entries = self.scan_templates()
        payload = [asdict(entry) for entry in entries]
        self.registry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.registry_path

    def load_registry(self) -> list[RegistryEntry]:
        """Loads registry entries from disk."""
        if not self.registry_path.exists():
            return self.scan_templates()
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.entries = [RegistryEntry(**entry) for entry in payload]
        return self.entries

    def find_best_match(
        self,
        style: Optional[str] = None,
        material: Optional[str] = None,
        minimum_quality: int = 0,
    ) -> Optional[RegistryEntry]:
        """Returns the best matching template entry."""
        entries = self.entries or self.load_registry()
        candidates = [
            entry
            for entry in entries
            if entry.quality >= minimum_quality
            and (style is None or entry.style == style)
            and (material is None or entry.material == material)
        ]
        return candidates[0] if candidates else None

    def filter_by_style(self, style: str) -> list[RegistryEntry]:
        """Filters registry entries by style."""
        entries = self.entries or self.load_registry()
        return [entry for entry in entries if entry.style == style]

    def filter_by_material(self, material: str) -> list[RegistryEntry]:
        """Filters registry entries by material."""
        entries = self.entries or self.load_registry()
        return [entry for entry in entries if entry.material == material]

    def filter_by_quality(self, minimum_quality: int) -> list[RegistryEntry]:
        """Filters registry entries by minimum quality score."""
        entries = self.entries or self.load_registry()
        return [entry for entry in entries if entry.quality >= minimum_quality]


__all__ = ["RegistryEntry", "TemplateRegistryBuilder"]
