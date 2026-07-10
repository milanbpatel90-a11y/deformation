"""Stage 4 — Mesh splitting and consistent renaming.

The detector in :mod:`factory.component_detector` flags each classified
island with ``is_separated_mesh``. This stage uses that flag to decide
whether to split the source mesh further:

* If every island already lives in its own Blender object, only renames
  are performed — no destructive ``separate_loose`` operator.
* Otherwise the merged object is split via ``bpy.ops.mesh.separate``
  (loose parts), then each new island is renamed.

After this stage the scene contains exactly the canonical objects::

    Frame
    LeftLens / RightLens
    LeftRim / RightRim
    Bridge
    LeftTemple / RightTemple
    TempleTips (optional)
    NosePads (optional)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import bpy  # type: ignore
    import bmesh  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import ClassifiedPart, ComponentClassification, PartKind
from .utils import get_logger

LOG = get_logger()


# Canonical names. These match the deformation engine's expected object
# names in ``backend/deformer/descriptor_loader.py``.
CANONICAL_NAMES: dict[PartKind, str] = {
    PartKind.FRAME: "Frame",
    PartKind.LEFT_LENS: "LeftLens",
    PartKind.RIGHT_LENS: "RightLens",
    PartKind.LEFT_RIM: "LeftRim",
    PartKind.RIGHT_RIM: "RightRim",
    PartKind.BRIDGE: "Bridge",
    PartKind.LEFT_TEMPLE: "LeftTemple",
    PartKind.RIGHT_TEMPLE: "RightTemple",
    PartKind.TEMPLE_TIPS: "TempleTips",
    PartKind.NOSE_PADS: "NosePads",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.splitter requires Blender.")


def split_merged_object(obj) -> int:
    """Split a single merged mesh into one object per loose island.

    Returns the number of new objects created (excluding the original).
    """
    require_blender()
    if obj is None or obj.type != "MESH":
        return 0

    # Make the target active and selected, then invoke ``separate``.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    before = set(o.name for o in bpy.data.objects)

    bpy.ops.mesh.separate(type="LOOSE")

    after = set(o.name for o in bpy.data.objects)
    new_names = sorted(after - before)
    return len(new_names)


def _find_object_by_name(name: str):
    require_blender()
    return bpy.data.objects.get(name)


def rename_part(part: ClassifiedPart) -> ClassifiedPart:
    """Rename the source Blender object to its canonical name."""
    require_blender()
    target_name = CANONICAL_NAMES.get(part.kind, part.name)
    src = _find_object_by_name(part.source_object)
    if src is None:
        LOG.warning("Source object %s missing — cannot rename.", part.source_object)
        return part

    # If a target-named object already exists, remove it to keep names unique.
    existing = _find_object_by_name(target_name)
    if existing is not None and existing.name != src.name:
        LOG.info("Removing stale %s before renaming %s.", existing.name, src.name)
        bpy.data.objects.remove(existing, do_unlink=True)

    src.name = target_name
    src.data.name = target_name
    part.name = target_name
    part.source_object = target_name
    return part


def _group_by_source(classification: ComponentClassification) -> dict[str, list[ClassifiedPart]]:
    bucket: dict[str, list[ClassifiedPart]] = {}
    for part in classification.parts:
        bucket.setdefault(part.source_object, []).append(part)
    return bucket


def _matches_world_bbox(obj, part: ClassifiedPart, tol: float = 1e-3) -> bool:
    """Check whether a Blender object's bbox matches a classified island."""
    if obj is None or obj.type != "MESH":
        return False
    if len(obj.data.vertices) != part.vertex_count:
        return False
    bbox = obj.data.bound_box
    mn = (bbox[0][0], bbox[0][1], bbox[0][2])
    mx = (bbox[6][0], bbox[6][1], bbox[6][2])
    for i in range(3):
        if abs(mn[i] - part.bbox.min[i]) > tol * max(1.0, abs(part.bbox.min[i])):
            return False
        if abs(mx[i] - part.bbox.max[i]) > tol * max(1.0, abs(part.bbox.max[i])):
            return False
    return True


def _resolve_target_object(part: ClassifiedPart, split_candidates: dict[str, list[str]]):
    """After a ``separate`` split, find the Blender object that corresponds
    to a classified island."""

    candidates = split_candidates.get(part.source_object, [])
    for name in candidates:
        obj = _find_object_by_name(name)
        if _matches_world_bbox(obj, part):
            return obj
    # Fallback: try the same name again.
    return _find_object_by_name(part.source_object)


def run_stage4(classification: ComponentClassification) -> ComponentClassification:
    """Split merged meshes and rename every part to its canonical name.

    The returned classification is a *new* object — the original is left
    untouched (useful for dry-run / preview flows).
    """
    require_blender()
    LOG.info("Stage 4 — splitting and renaming.")

    # 1) Split every merged object into loose islands.
    split_log: dict[str, list[str]] = {}
    grouped = _group_by_source(classification)
    for source_name, parts in grouped.items():
        if len(parts) <= 1 and parts[0].is_separated_mesh:
            LOG.debug("Source %s already split — skipping separate.", source_name)
            continue
        if len(parts) <= 1 and parts[0].kind != PartKind.FRAME:
            # Only one island and it isn't the frame — nothing to split.
            LOG.debug("Source %s has a single non-frame island — skipping.", source_name)
            continue
        obj = _find_object_by_name(source_name)
        new_count = split_merged_object(obj)
        if new_count == 0:
            LOG.debug("Separate produced no new objects for %s.", source_name)
            continue
        # Record all current scene object names that share this source.
        all_names = [o.name for o in bpy.data.objects]
        split_log[source_name] = all_names
        LOG.info("Split %s → +%d objects", source_name, new_count)

    # 2) Update ``source_object`` references for islands whose source was split.
    for part in classification.parts:
        if part.source_object in split_log:
            target = _resolve_target_object(part, split_log)
            if target is not None:
                part.source_object = target.name

    # 3) Rename everything.
    for part in classification.parts:
        rename_part(part)

    LOG.info("Stage 4 done — canonical object names assigned.")
    return classification


def ensure_only_mesh_objects(keep: Iterable[str] | None = None) -> None:
    """Final cleanup: drop any remaining non-mesh objects."""
    require_blender()
    keep_set = set(keep or ())
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            continue
        if obj.name in keep_set:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
