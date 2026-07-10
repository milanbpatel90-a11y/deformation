"""Stage 6 — Generate descriptor.json compatible with the deformation engine.

The output schema matches the payload validated by
``backend/deformer/descriptor_loader.DescriptorLoader``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    ClassifiedPart,
    ComponentClassification,
    PartKind,
    VertexGroupSpec,
)
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.descriptor_generator requires Blender.")


@dataclass(slots=True)
class DescriptorBuildResult:
    """Container holding the built descriptor and the output path."""

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


def _compute_hinge_pivot(obj, side: str) -> tuple[float, float, float]:
    """Approximate the hinge pivot from the temple mesh.

    The hinge sits at the extreme X of the temple (inner end).
    """
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) == 0:
        return (0.0, 0.0, 0.0)
    coords = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    if side == "left":
        x_idx = int(np.argmin(coords[:, 0]))
    else:
        x_idx = int(np.argmax(coords[:, 0]))
    pivot = coords[x_idx]
    return (float(pivot[0]), float(pivot[1]), float(pivot[2]))


def _compute_temple_axis(obj, pivot: tuple[float, float, float], side: str) -> tuple[float, float, float]:
    """Estimate the temple axis vector by fitting a line to the temple mesh."""
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) < 3:
        return (-1.0, 0.0, 0.0) if side == "left" else (1.0, 0.0, 0.0)
    coords = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    pivot_arr = np.array(pivot, dtype=np.float64)
    rel = coords - pivot_arr
    # Use the principal axis of the temple (PCA).
    try:
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        axis = vh[0]
        # Point away from the hinge (temples extend backward).
        if side == "left" and axis[0] > 0:
            axis = -axis
        if side == "right" and axis[0] < 0:
            axis = -axis
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return (-1.0, 0.0, 0.0) if side == "left" else (1.0, 0.0, 0.0)
        return tuple((axis / norm).tolist())
    except np.linalg.LinAlgError:
        return (-1.0, 0.0, 0.0) if side == "left" else (1.0, 0.0, 0.0)


def _fit_plane_normal(points: np.ndarray) -> tuple[float, float, float]:
    if points.size == 0 or len(points) < 3:
        return (0.0, 0.0, 1.0)
    centered = points.astype(np.float64) - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return (0.0, 0.0, 1.0)
    if normal[2] < 0:
        normal = -normal
    return tuple((normal / norm).tolist())


def _build_hinges(classification: ComponentClassification) -> dict[str, Any]:
    left_temple = _find_object(classification.left_temple_part or "LeftTemple")
    right_temple = _find_object(classification.right_temple_part or "RightTemple")

    left_pivot = _compute_hinge_pivot(left_temple, "left")
    right_pivot = _compute_hinge_pivot(right_temple, "right")

    left_axis = _compute_temple_axis(left_temple, left_pivot, "left")
    right_axis = _compute_temple_axis(right_temple, right_pivot, "right")

    return {
        "left": {
            "part": "LeftTemple",
            "pivot": left_pivot,
            "axis": left_axis,
            "confidence": 1.0,
        },
        "right": {
            "part": "RightTemple",
            "pivot": right_pivot,
            "axis": right_axis,
            "confidence": 1.0,
        },
    }


def _build_bridge(classification: ComponentClassification) -> dict[str, Any]:
    bridge_obj = _find_object(classification.bridge_part or "Bridge")
    if bridge_obj is None:
        return {
            "part": "Bridge",
            "type": "pad",
            "width_mm": 16.0,
            "confidence": 0.3,
        }
    bbox = _bbox_of_object(bridge_obj)
    if bbox:
        width_mm = (bbox.max[0] - bbox.min[0]) * 1000.0
    else:
        width_mm = 16.0
    return {
        "part": "Bridge",
        "type": "pad",
        "width_mm": round(float(width_mm), 2),
        "confidence": 1.0,
    }


def _build_rim_loops(classification: ComponentClassification) -> dict[str, Any]:
    frame_obj = _find_object("Frame")
    left_rim_obj = _find_object("LeftRim")
    right_rim_obj = _find_object("RightRim")

    # If separate rim objects exist, use those; otherwise the frame itself.
    left_part = "LeftRim" if left_rim_obj else "Frame"
    right_part = "RightRim" if right_rim_obj else "Frame"

    return {
        "frame": {"part": "Frame", "confidence": 1.0},
        "left_lens": {"part": left_part, "confidence": 1.0},
        "right_lens": {"part": right_part, "confidence": 1.0},
    }


def _build_lens_planes(classification: ComponentClassification) -> dict[str, Any]:
    left_lens = _find_object("LeftLens")
    right_lens = _find_object("RightLens")

    def lens_data(obj, side: str) -> dict[str, Any]:
        if obj is None or obj.type != "MESH" or len(obj.data.vertices) == 0:
            return {
                "part": f"{side.capitalize()}Lens",
                "aspect_width_mm": 50.0,
                "aspect_height_mm": 46.0,
            }
        bbox = _bbox_of_object(obj)
        if bbox:
            w_mm = (bbox.max[0] - bbox.min[0]) * 1000.0
            h_mm = (bbox.max[1] - bbox.min[1]) * 1000.0
        else:
            w_mm, h_mm = 50.0, 46.0
        coords = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
        origin = tuple(coords.mean(axis=0).tolist())
        normal = _fit_plane_normal(coords)
        return {
            "part": f"{side.capitalize()}Lens",
            "origin": origin,
            "normal": normal,
            "aspect_width_mm": round(float(w_mm), 2),
            "aspect_height_mm": round(float(h_mm), 2),
        }

    return {
        "left": lens_data(left_lens, "left"),
        "right": lens_data(right_lens, "right"),
    }


def _build_symmetry_plane(classification: ComponentClassification) -> dict[str, Any]:
    frame_obj = _find_object("Frame")
    width_mm = 140.0
    if frame_obj and frame_obj.type == "MESH":
        bbox = _bbox_of_object(frame_obj)
        if bbox:
            width_mm = (bbox.max[0] - bbox.min[0]) * 1000.0
    return {
        "axis": "X",
        "frame_width_mm": round(float(width_mm), 2),
        "score": 1.0,
    }


def _build_temple_pivots(classification: ComponentClassification) -> dict[str, Any]:
    left_temple = _find_object("LeftTemple")
    right_temple = _find_object("RightTemple")

    def temple_data(obj, side: str) -> dict[str, Any]:
        if obj is None or obj.type != "MESH" or len(obj.data.vertices) == 0:
            return {"part": f"{side.capitalize()}Temple", "estimated_length_mm": 135.0}
        bbox = _bbox_of_object(obj)
        if bbox:
            length_mm = (bbox.max[0] - bbox.min[0]) * 1000.0
        else:
            length_mm = 135.0
        return {
            "part": f"{side.capitalize()}Temple",
            "estimated_length_mm": round(float(length_mm), 2),
        }

    return {
        "left": temple_data(left_temple, "left"),
        "right": temple_data(right_temple, "right"),
    }


def _build_deformation_regions(classification: ComponentClassification) -> dict[str, Any]:
    """Region hints for the deformation engine."""
    return {
        "bridge": {"type": "pad"},
        "frame": {"family": "geometric"},
        "temples": {"average_length_mm": 135.0},
    }


def _build_vertex_groups(classification: ComponentClassification) -> dict[str, list[str]]:
    """Map logical VG names → canonical object names (as expected by the loader)."""
    return {
        "frame": ["Frame"],
        "bridge": ["Bridge"],
        "left_rim": ["LeftRim"] if _find_object("LeftRim") else ["Frame"],
        "right_rim": ["RightRim"] if _find_object("RightRim") else ["Frame"],
        "left_lens": ["LeftLens"],
        "right_lens": ["RightLens"],
        "left_temple": ["LeftTemple"],
        "right_temple": ["RightTemple"],
        "nose_pads": ["NosePads"] if _find_object("NosePads") else [],
        "temple_tips": ["TempleTips"] if _find_object("TempleTips") else [],
    }


def run_stage6(
    template_id: str,
    classification: ComponentClassification,
    analysis_bbox: BoundingBox,
    output_dir: Path,
) -> DescriptorBuildResult:
    """Generate descriptor.json and write it to ``output_dir``."""
    require_blender()
    LOG.info("Stage 6 — generating descriptor for %s.", template_id)

    payload = {
        "hinges": _build_hinges(classification),
        "bridge": _build_bridge(classification),
        "rim_loops": _build_rim_loops(classification),
        "lens_planes": _build_lens_planes(classification),
        "symmetry_plane": _build_symmetry_plane(classification),
        "temple_pivots": _build_temple_pivots(classification),
        "deformation_regions": _build_deformation_regions(classification),
        "vertex_groups": _build_vertex_groups(classification),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{template_id}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    LOG.info("Stage 6 done — wrote %s", output_path)

    return DescriptorBuildResult(payload=payload, output_path=output_path)
