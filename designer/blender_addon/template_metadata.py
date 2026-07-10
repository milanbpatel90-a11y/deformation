from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .utils import ensure_directory, get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TemplateMetadata:
    """Serializable template metadata."""

    id: str
    family: str
    material: str
    rim: str
    bridge: str
    frame_width: float
    lens_width: float
    lens_height: float
    bridge_width: float
    temple_length: float
    triangle_count: int
    vertex_count: int
    vertex_groups: bool
    uv: bool
    validated: bool
    quality_score: int
    style: str = ""
    frame_height: float = 0.0
    frame_thickness: float = 0.0
    validation_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Returns metadata as a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Returns metadata as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> Path:
        """Writes metadata JSON to disk."""
        target = Path(path).expanduser().resolve()
        ensure_directory(target.parent)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemplateMetadata":
        """Creates metadata from a dictionary."""
        return cls(**payload)

    @classmethod
    def from_json(cls, payload: str) -> "TemplateMetadata":
        """Creates metadata from a JSON string."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def read_json(cls, path: str | Path) -> "TemplateMetadata":
        """Loads metadata from a JSON file."""
        source = Path(path).expanduser().resolve()
        return cls.from_json(source.read_text(encoding="utf-8"))


__all__ = ["TemplateMetadata"]
