from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_directory

try:
    import bpy
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Object = object  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ThumbnailSet:
    """Resolved thumbnail output paths."""

    front: Path
    side: Path
    perspective: Path


def render_thumbnails(template_id: str, output_dir: str | Path) -> ThumbnailSet:
    """Resolves thumbnail targets for a template.

    Real rendering can be expanded in a later milestone without changing callers.
    """
    base_dir = ensure_directory(output_dir)
    return ThumbnailSet(
        front=base_dir / f"{template_id}_front.png",
        side=base_dir / f"{template_id}_side.png",
        perspective=base_dir / f"{template_id}_perspective.png",
    )

