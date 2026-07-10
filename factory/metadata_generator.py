"""Stage 7 — Generate metadata.json with eyewear classification.

Outputs:

* shape (rectangle, round, cat_eye, etc.)
* material (acetate, metal, titanium, etc.)
* rim_type (full_rim, semi_rimless, rimless)
* frame_family (acetate, metal, rimless, mixed, sport)
* bridge_type (pad, keyhole, single, double)
* frame_width, bridge_width, lens_width, lens_height, temple_length (mm)
* triangle_count, vertex_count
* quality_score (0–100)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import bpy  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import (
    BoundingBox,
    BridgeType,
    ClassifiedPart,
    ComponentClassification,
    FrameFamily,
    Material,
    PartKind,
    RimType,
    Shape,
)
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.metadata_generator requires Blender.")


@dataclass(slots=True)
class MetadataBuildResult:
    payload: dict[str, Any]
    output_path: Path


def _find_object(name: str):
    require_blender()
    return bpy.data.objects.get(name)


def _bbox_of_object(obj) -> BoundingBox | None:
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) == 0:
        return None
    bbox = obj.data.bound_box
    mn = (bbox[0][0], bbox[0][1], bbox[0][2])
    mx = (bbox[6][0], bbox[6][1], bbox[6][2])
    return BoundingBox(min=mn, max=mx)


def _world_bbox_of_all() -> BoundingBox:
    """Combined world-space bbox of every mesh."""
    require_blender()
    all_coords: list[np.ndarray] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            eval_obj.to_mesh_clear()
            continue
        mw = obj.matrix_world
        coords = np.array(
            [(mw @ v.co)[:] for v in mesh.vertices],
            dtype=np.float64,
        )
        all_coords.append(coords)
        eval_obj.to_mesh_clear()
    if not all_coords:
        return BoundingBox(min=(0, 0, 0), max=(0, 0, 0))
    pts = np.concatenate(all_coords, axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return BoundingBox(
        min=(float(mn[0]), float(mn[1]), float(mn[2])),
        max=(float(mx[0]), float(mx[1]), float(mx[2])),
    )


def _count_total_triangles() -> int:
    require_blender()
    tris = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        try:
            obj.data.calc_loop_triangles()
            tris += len(obj.data.loop_triangles)
        except Exception:
            tris += len(obj.data.polygons) * 2
    return tris


def _count_total_vertices() -> int:
    require_blender()
    return sum(len(obj.data.vertices) for obj in bpy.data.objects if obj.type == "MESH")


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------

def _classify_shape(classification: ComponentClassification) -> Shape:
    """Heuristic shape classification from bbox aspect ratios."""
    frame_obj = _find_object("Frame")
    if frame_obj is None:
        return Shape.OTHER
    bbox = _bbox_of_object(frame_obj)
    if bbox is None:
        return Shape.OTHER
    w = (bbox.max[0] - bbox.min[0]) * 1000.0
    h = (bbox.max[1] - bbox.min[1]) * 1000.0
    if w <= 0 or h <= 0:
        return Shape.OTHER
    aspect = h / w

    # Check for cat-eye: upswept outer corners
    left_lens = _find_object("LeftLens")
    right_lens = _find_object("RightLens")
    cat_eye = False
    for obj in (left_lens, right_lens):
        if obj and obj.type == "MESH":
            coords = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
            top = coords[coords[:, 1] > np.percentile(coords[:, 1], 75)]
            if len(top) > 10:
                outer_x = np.mean(top[:, 0])
                center_x = np.mean(coords[:, 0])
                if (obj.name == "LeftLens" and outer_x > center_x) or (
                    obj.name == "RightLens" and outer_x < center_x
                ):
                    cat_eye = True

    if cat_eye:
        return Shape.CAT_EYE
    if aspect > 0.95:
        return Shape.ROUND if aspect > 1.1 else Shape.SQUARE
    if aspect > 0.7:
        return Shape.OVAL
    if aspect < 0.55:
        return Shape.RECTANGLE
    return Shape.OTHER


def _classify_material(classification: ComponentClassification) -> Material:
    """Heuristic material from mesh density and material slots."""
    # Check material slot names.
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if not slot.material:
                continue
            name = slot.material.name.lower()
            if any(k in name for k in ("metal", "steel", "titanium", "chrome", "gold", "silver")):
                return Material.METAL
            if any(k in name for k in ("acetate", "plastic", "tortoise", "tort", "pc", "tr90")):
                return Material.ACETATE
            if "carbon" in name:
                return Material.CARBON
            if "wood" in name:
                return Material.WOOD

    # Fallback: triangle density → metal frames are usually thinner/more tri-dense per mm.
    tris = _count_total_triangles()
    bbox = _world_bbox_of_all()
    vol = max(bbox.size[0], 1e-6) * max(bbox.size[1], 1e-6) * max(bbox.size[2], 1e-6)
    density = tris / vol if vol > 0 else 0
    if density > 5e6:
        return Material.METAL
    return Material.ACETATE


def _classify_rim_type(classification: ComponentClassification) -> RimType:
    """Detect full-rim / semi-rimless / rimless."""
    has_lens = _find_object("LeftLens") is not None or _find_object("RightLens") is not None
    has_rim = _find_object("LeftRim") is not None or _find_object("RightRim") is not None
    has_frame = _find_object("Frame") is not None

    if has_frame and has_rim:
        return RimType.FULL_RIM
    if has_frame and not has_rim:
        # Frame exists but no separate rims → could be semi-rimless
        return RimType.SEMI_RIMLESS
    return RimType.RIMLESS


def _classify_frame_family(
    material: Material,
    rim_type: RimType,
    shape: Shape,
) -> FrameFamily:
    if rim_type == RimType.RIMLESS:
        return FrameFamily.RIMLESS
    if material in (Material.METAL, Material.TITANIUM):
        return FrameFamily.METAL
    if material == Material.ACETATE:
        return FrameFamily.ACETATE
    if shape == Shape.CAT_EYE:
        return FrameFamily.ACETATE
    return FrameFamily.MIXED


def _classify_bridge_type(classification: ComponentClassification) -> BridgeType:
    """Heuristic bridge type from geometry."""
    bridge = _find_object("Bridge")
    if bridge is None:
        return BridgeType.UNIVERSAL

    bbox = _bbox_of_object(bridge)
    if bbox is None:
        return BridgeType.UNIVERSAL

    w = (bbox.max[0] - bbox.min[0]) * 1000.0
    h = (bbox.max[1] - bbox.min[1]) * 1000.0
    d = (bbox.max[2] - bbox.min[2]) * 1000.0

    # Keyhole: tall, narrow, significant depth (curved cutout)
    if h > 2.0 * w and d > 3.0:
        return BridgeType.KEYHOLE
    # Pad bridge: wide, low, two distinct pads
    if w > 3.0 * h:
        return BridgeType.PAD
    # Double bridge: two parallel bars
    if h > 1.5 * w and w > 2.0:
        return BridgeType.DOUBLE
    return BridgeType.SINGLE


def _measure_dimensions(classification: ComponentClassification) -> dict[str, float]:
    """Extract all key dimensions in mm."""
    dims: dict[str, float] = {}

    frame = _find_object("Frame")
    if frame and (bbox := _bbox_of_object(frame)):
        dims["frame_width"] = round((bbox.max[0] - bbox.min[0]) * 1000.0, 2)
        dims["frame_height"] = round((bbox.max[1] - bbox.min[1]) * 1000.0, 2)

    left_lens = _find_object("LeftLens")
    if left_lens and (bbox := _bbox_of_object(left_lens)):
        dims["lens_width"] = round((bbox.max[0] - bbox.min[0]) * 1000.0, 2)
        dims["lens_height"] = round((bbox.max[1] - bbox.min[1]) * 1000.0, 2)

    right_lens = _find_object("RightLens")
    if right_lens and (bbox := _bbox_of_object(right_lens)):
        # Average both lenses
        lw = dims.get("lens_width", 0)
        lh = dims.get("lens_height", 0)
        dims["lens_width"] = round((lw + (bbox.max[0] - bbox.min[0]) * 1000.0) / 2, 2)
        dims["lens_height"] = round((lh + (bbox.max[1] - bbox.min[1]) * 1000.0) / 2, 2)

    bridge = _find_object("Bridge")
    if bridge and (bbox := _bbox_of_object(bridge)):
        dims["bridge_width"] = round((bbox.max[0] - bbox.min[0]) * 1000.0, 2)

    left_temple = _find_object("LeftTemple")
    right_temple = _find_object("RightTemple")
    lengths = []
    for t in (left_temple, right_temple):
        if t and (bbox := _bbox_of_object(t)):
            lengths.append((bbox.max[0] - bbox.min[0]) * 1000.0)
    if lengths:
        dims["temple_length"] = round(sum(lengths) / len(lengths), 2)

    return dims


def _generate_id(template_name: str) -> str:
    """Generate a stable short ID from the template name."""
    import hashlib
    h = hashlib.md5(template_name.encode()).hexdigest()[:8]
    base = "".join(c.lower() for c in template_name if c.isalnum())[:12]
    return f"{base}_{h}"


def run_stage7(
    template_id: str,
    template_name: str,
    classification: ComponentClassification,
    analysis_bbox: BoundingBox,
    quality_score: int,
    output_dir: Path,
) -> MetadataBuildResult:
    """Generate metadata.json and write it to ``output_dir``."""
    require_blender()
    LOG.info("Stage 7 — generating metadata for %s.", template_id)

    shape = _classify_shape(classification)
    material = _classify_material(classification)
    rim_type = _classify_rim_type(classification)
    frame_family = _classify_frame_family(material, rim_type, shape)
    bridge_type = _classify_bridge_type(classification)
    dims = _measure_dimensions(classification)

    payload = {
        "id": template_id,
        "name": template_name,
        "shape": shape.value,
        "material": material.value,
        "rim_type": rim_type.value,
        "frame_family": frame_family.value,
        "bridge_type": bridge_type.value,
        "dimensions": dims,
        "triangle_count": _count_total_triangles(),
        "vertex_count": _count_total_vertices(),
        "quality_score": quality_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": template_name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{template_id}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    LOG.info("Stage 7 done — wrote %s", output_path)

    return MetadataBuildResult(payload=payload, output_path=output_path)
