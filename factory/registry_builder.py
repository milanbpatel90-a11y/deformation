"""Stage 10 — Registry builder.

Maintains ``templates/registry.json`` as a flat array of template entries.
Each entry contains:

* id
* shape
* material
* frame_family
* bridge_type
* rim_type
* descriptor (relative path)
* metadata (relative path)
* thumbnail (relative path)
* quality
* processed_glb (relative path)
* passed (bool)
* created_at (ISO8601)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import RegistryEntry
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.registry_builder requires Blender.")


def load_registry(registry_path: Path) -> list[dict[str, Any]]:
    """Load existing registry or return empty list."""
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text())
        except Exception as e:
            LOG.warning("Failed to parse %s: %s", registry_path, e)
    return []


def save_registry(registry_path: Path, entries: list[dict[str, Any]]) -> None:
    """Atomically write registry."""
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(registry_path)
    LOG.info("Registry saved: %d entries → %s", len(entries), registry_path)


def _rel_path(registry_root: Path, path: Path) -> str:
    """Return path relative to registry root, using POSIX separators."""
    try:
        return path.relative_to(registry_root).as_posix()
    except ValueError:
        return path.name


def upsert_entry(
    registry_path: Path,
    template_id: str,
    metadata: dict[str, Any],
    descriptor_rel: str,
    metadata_rel: str,
    thumbnail_rel: str,
    quality: int,
    processed_glb_rel: str,
    passed: bool,
) -> None:
    """Insert or update a single registry entry."""
    entries = load_registry(registry_path)

    # Remove any existing entry with the same id.
    entries = [e for e in entries if e.get("id") != template_id]

    entry = {
        "id": template_id,
        "shape": metadata.get("shape", "other"),
        "material": metadata.get("material", "unknown"),
        "frame_family": metadata.get("frame_family", "mixed"),
        "bridge_type": metadata.get("bridge_type", "universal"),
        "rim_type": metadata.get("rim_type", "full_rim"),
        "descriptor": descriptor_rel,
        "metadata": metadata_rel,
        "thumbnail": thumbnail_rel,
        "quality": int(quality),
        "processed_glb": processed_glb_rel,
        "passed": bool(passed),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    entries.append(entry)
    save_registry(registry_path, entries)


def remove_entry(registry_path: Path, template_id: str) -> bool:
    """Remove an entry by id. Returns True if removed."""
    entries = load_registry(registry_path)
    before = len(entries)
    entries = [e for e in entries if e.get("id") != template_id]
    if len(entries) < before:
        save_registry(registry_path, entries)
        return True
    return False


def list_entries(registry_path: Path) -> list[dict[str, Any]]:
    """Return all registry entries (read-only)."""
    return load_registry(registry_path)


def find_by_id(registry_path: Path, template_id: str) -> dict[str, Any] | None:
    for e in load_registry(registry_path):
        if e.get("id") == template_id:
            return e
    return None


def find_by_shape(registry_path: Path, shape: str) -> list[dict[str, Any]]:
    return [e for e in load_registry(registry_path) if e.get("shape") == shape]


def find_by_material(registry_path: Path, material: str) -> list[dict[str, Any]]:
    return [e for e in load_registry(registry_path) if e.get("material") == material]


def run_stage10(
    template_id: str,
    template_name: str,
    metadata: dict[str, Any],
    quality: int,
    passed: bool,
    descriptor_path: Path,
    metadata_path: Path,
    thumbnail_path: Path,
    processed_glb_path: Path,
    templates_root: Path,
) -> None:
    """Convenience wrapper: compute relative paths and upsert."""
    require_blender()
    LOG.info("Stage 10 — updating registry for %s.", template_id)

    # Thumbnail: pick the front view as the canonical thumbnail.
    thumb_rel = _rel_path(templates_root, thumbnail_path)
    if "_front" not in thumb_rel and "_angle" not in thumb_rel:
        # If we have the front/side/angle triplet, prefer front.
        front_path = thumbnail_path.with_name(f"{template_id}_front.png")
        if front_path.exists():
            thumb_rel = _rel_path(templates_root, front_path)

    upsert_entry(
        registry_path=templates_root / "registry.json",
        template_id=template_id,
        metadata=metadata,
        descriptor_rel=_rel_path(templates_root, descriptor_path),
        metadata_rel=_rel_path(templates_root, metadata_path),
        thumbnail_rel=thumb_rel,
        quality=quality,
        processed_glb_rel=_rel_path(templates_root, processed_glb_path),
        passed=passed,
    )
