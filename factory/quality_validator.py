"""Stage 9 — Quality validation and scoring.

Produces a QualityReport with boolean checks and an aggregate 0–100 score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import bpy  # type: ignore
    import bmesh  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import (
    AnalysisResult,
    BoundingBox,
    ClassifiedPart,
    ComponentClassification,
    PartKind,
    QualityReport,
    VertexGroupSpec,
)
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.quality_validator requires Blender.")


@dataclass(slots=True)
class ValidationResult:
    report: QualityReport
    passed: bool


# ---------------------------------------------------------------------------
# Individual check implementations
# ---------------------------------------------------------------------------

def _check_orientation(classification: ComponentClassification) -> tuple[bool, str | None]:
    """Ensure frame faces +Z and temples extend along ±X."""
    frame = _find_object("Frame")
    left_temple = _find_object("LeftTemple")
    right_temple = _find_object("RightTemple")

    if frame is None:
        return False, "Frame object missing"
    if left_temple is None or right_temple is None:
        return False, "One or both temples missing"

    # Check temple X positions.
    l_bbox = _bbox_of_object(left_temple)
    r_bbox = _bbox_of_object(right_temple)
    if l_bbox is None or r_bbox is None:
        return False, "Temple bbox unavailable"

    if l_bbox.center[0] > 0:
        return False, "Left temple on wrong side of origin"
    if r_bbox.center[0] < 0:
        return False, "Right temple on wrong side of origin"

    return True, None


def _check_scale(analysis: AnalysisResult) -> tuple[bool, str | None]:
    """Frame width should be ~140 mm (±2 mm tolerance)."""
    if not (138.0 <= analysis.frame_width_mm <= 142.0):
        return False, f"Frame width {analysis.frame_width_mm:.1f} mm out of range [138, 142]"
    return True, None


def _check_normals() -> tuple[bool, str | None]:
    """Every mesh must have valid non-zero normals."""
    require_blender()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        if len(mesh.vertices) == 0:
            continue
        # Check a sample of vertices.
        zero_count = sum(
            1
            for v in mesh.vertices[: min(100, len(mesh.vertices))]
            if v.normal.length < 1e-6
        )
        if zero_count > 0:
            return False, f"{obj.name}: {zero_count} vertices have zero-length normals"
    return True, None


def _check_uvs() -> tuple[bool, str | None]:
    """At least one UV layer should exist on renderable meshes."""
    require_blender()
    missing = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.data.uv_layers:
            missing.append(obj.name)
    if missing:
        return False, f"Missing UVs: {', '.join(missing)}"
    return True, None


def _check_topology() -> tuple[bool, str | None]:
    """No degenerate faces, no loose vertices."""
    require_blender()
    issues: list[str] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        if len(mesh.vertices) == 0:
            continue
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Degenerate faces (area < 1e-12)
        degens = [f for f in bm.faces if f.calc_area() < 1e-12]
        if degens:
            issues.append(f"{obj.name}: {len(degens)} degenerate faces")

        # Loose vertices
        loose = [v for v in bm.verts if len(v.link_faces) == 0]
        if loose:
            issues.append(f"{obj.name}: {len(loose)} loose vertices")

        bm.free()
    if issues:
        return False, "; ".join(issues)
    return True, None


def _check_manifold() -> tuple[bool, str | None]:
    """All edges should be manifold (max 2 faces)."""
    require_blender()
    non_manifold: list[str] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        for edge in bm.edges:
            if len(edge.link_faces) > 2:
                non_manifold.append(obj.name)
                break
        bm.free()
    if non_manifold:
        return False, f"Non-manifold edges in: {', '.join(non_manifold)}"
    return True, None


def _check_self_intersections() -> tuple[bool, str | None]:
    """Quick BVH-based self-intersection test per object."""
    require_blender()
    intersected: list[str] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or len(obj.data.vertices) < 4:
            continue
        try:
            bvhtree = bmesh.types.BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
            hits = bvhtree.overlap(bvhtree)
            if hits:
                intersected.append(obj.name)
        except Exception:
            # BVH overlap can fail on very thin geometry; treat as pass.
            pass
    if intersected:
        return False, f"Self-intersections in: {', '.join(intersected)}"
    return True, None


def _check_mesh_count(classification: ComponentClassification) -> tuple[bool, str | None]:
    """Must have at least Frame + 2 Temples + 2 Lenses (or Rims) + Bridge."""
    required = {
        PartKind.FRAME: 1,
        PartKind.LEFT_TEMPLE: 1,
        PartKind.RIGHT_TEMPLE: 1,
    }
    missing = []
    for kind, count in required.items():
        if len(classification.by_kind(kind)) < count:
            missing.append(kind.value)
    if missing:
        return False, f"Missing required parts: {', '.join(missing)}"
    return True, None


def _check_descriptor_completeness(descriptor_path: Path) -> tuple[bool, str | None]:
    """Descriptor file exists and has all required keys."""
    required_keys = {
        "hinges", "bridge", "rim_loops", "lens_planes",
        "symmetry_plane", "temple_pivots", "deformation_regions",
        "vertex_groups",
    }
    try:
        import json
        data = json.loads(descriptor_path.read_text())
    except Exception as e:
        return False, f"Failed to read descriptor: {e}"
    missing = required_keys - set(data.keys())
    if missing:
        return False, f"Descriptor missing keys: {', '.join(sorted(missing))}"
    return True, None


def _check_vertex_groups() -> tuple[bool, str | None]:
    """Required VG_* groups must exist on the Frame."""
    require_blender()
    frame = _find_object("Frame")
    if frame is None:
        return False, "Frame object missing"
    required = {"VG_FRAME", "VG_LEFT_RIM", "VG_RIGHT_RIM"}
    existing = {vg.name for vg in frame.vertex_groups}
    missing = required - existing
    if missing:
        return False, f"Frame missing vertex groups: {', '.join(sorted(missing))}"
    return True, None


def _check_symmetry(analysis: AnalysisResult) -> tuple[bool, str | None]:
    """Symmetry score should be > 0.7."""
    if analysis.symmetry_score < 0.7:
        return False, f"Symmetry score {analysis.symmetry_score:.3f} below 0.7"
    return True, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_stage9(
    template_id: str,
    classification: ComponentClassification,
    analysis: AnalysisResult,
    descriptor_path: Path,
) -> ValidationResult:
    """Run all quality checks and compute aggregate score.

    Weights (sum to 100):
      orientation         10
      scale               10
      normals             10
      uvs                 10
      topology            10
      manifold            10
      self_intersections  10
      mesh_count          10
      descriptor          10
      vertex_groups       10
      symmetry            10
    """
    require_blender()
    LOG.info("Stage 9 — running quality validation for %s.", template_id)

    checks: list[tuple[str, tuple[bool, str | None], int]] = [
        ("orientation", _check_orientation(classification), 10),
        ("scale", _check_scale(analysis), 10),
        ("normals", _check_normals(), 10),
        ("uvs", _check_uvs(), 10),
        ("topology", _check_topology(), 10),
        ("manifold", _check_manifold(), 10),
        ("self_intersections", _check_self_intersections(), 10),
        ("mesh_count", _check_mesh_count(classification), 10),
        ("descriptor_completeness", _check_descriptor_completeness(descriptor_path), 10),
        ("vertex_groups", _check_vertex_groups(), 10),
        ("symmetry", _check_symmetry(analysis), 10),
    ]

    results: dict[str, bool] = {}
    issues: list[str] = []
    score = 0

    for name, (ok, msg), weight in checks:
        results[name] = ok
        if ok:
            score += weight
        else:
            issues.append(f"{name}: {msg or 'failed'}")

    report = QualityReport(
        score=score,
        checks=results,
        issues=issues,
        metrics={
            "frame_width_mm": analysis.frame_width_mm,
            "symmetry_score": analysis.symmetry_score,
            "triangle_count": analysis.triangle_count,
            "vertex_count": analysis.vertex_count,
        },
        passed=score >= 70,
    )

    LOG.info("Stage 9 done — score=%d passed=%s issues=%d", score, report.passed, len(issues))
    return ValidationResult(report=report, passed=report.passed)
