"""Stage 2 — Geometric and topological analysis of the imported scene."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import bpy  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import AnalysisResult, BoundingBox
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.analyzer requires Blender's Python runtime.")


def combined_world_points() -> np.ndarray:
    """Return Nx3 array of every world-space vertex across all mesh objects."""
    require_blender()
    pieces: list[np.ndarray] = []
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
        pieces.append(coords)
        eval_obj.to_mesh_clear()
    if not pieces:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def compute_bbox(points: np.ndarray) -> BoundingBox:
    if points.size == 0:
        return BoundingBox(min=(0.0, 0.0, 0.0), max=(0.0, 0.0, 0.0))
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return BoundingBox(
        min=(float(mn[0]), float(mn[1]), float(mn[2])),
        max=(float(mx[0]), float(mx[1]), float(mx[2])),
    )


def count_connected_components(points: np.ndarray, voxel: float = 0.5) -> int:
    """Approximate connected-component count via voxel hashing.

    Voxel size is expressed in millimetres-equivalent; pass ``points`` in the
    same unit. 0.5 mm is small enough to split clearly separated mesh islands
    while tolerating floating-point drift.
    """

    if points.size == 0:
        return 0
    voxel = max(float(voxel), 1e-4)
    keys: set[tuple[int, int, int]] = set()
    for p in points:
        keys.add((int(round(p[0] / voxel)), int(round(p[1] / voxel)), int(round(p[2] / voxel))))
    if not keys:
        return 0
    # Flood-fill 26-neighbour connectivity over the voxel grid.
    visited: set[tuple[int, int, int]] = set()
    components = 0
    key_to_set: dict[tuple[int, int, int], int] = {k: i for i, k in enumerate(sorted(keys))}
    parent: list[int] = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pre-compute neighbour offsets once.
    neigh = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neigh.append((dx, dy, dz))

    key_list = list(key_to_set.keys())
    index_of = key_to_set
    for i, k in enumerate(key_list):
        if k in visited:
            continue
        stack = [k]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cx, cy, cz = cur
            for dx, dy, dz in neigh:
                nk = (cx + dx, cy + dy, cz + dz)
                j = index_of.get(nk)
                if j is not None:
                    union(i, j)
                    stack.append(nk)
        components += 1
    # Recount unique roots.
    return len({find(x) for x in range(len(keys))})


def symmetry_score(points: np.ndarray, axis: str = "x") -> float:
    """Return [0..1] score measuring left/right (or top/bottom) symmetry.

    ``axis`` selects the plane normal: ``x`` → mirror across YZ-plane,
    ``y`` → mirror across XZ-plane.
    """
    if points.size == 0 or len(points) < 8:
        return 0.0
    idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    axis_vals = points[:, idx]
    mirror_vals = -axis_vals
    # Combine both halves so we can compare point counts and centroid offset.
    flipped = points.copy()
    flipped[:, idx] = mirror_vals
    # Quantise to nearest 0.5 mm and compare sets.
    def quantise(arr: np.ndarray) -> set[tuple[int, int, int]]:
        return {
            (int(round(p[0] * 2)), int(round(p[1] * 2)), int(round(p[2] * 2)))
            for p in arr
        }
    a = quantise(points)
    b = quantise(flipped)
    if not a or not b:
        return 0.0
    intersect = len(a & b)
    union = max(len(a | b), 1)
    return float(intersect) / float(union)


def count_topology() -> tuple[int, int, int, int, int]:
    """Return (objects, meshes, vertices, edges, triangles)."""
    require_blender()
    obj_count = len(bpy.data.objects)
    mesh_count = 0
    verts = 0
    edges = 0
    tris = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh_count += 1
        verts += len(obj.data.vertices)
        edges += len(obj.data.edges)
        # ``calc_loop_triangles`` works in 3.x; fall back to polygon*3.
        try:
            obj.data.calc_loop_triangles()
            tris += len(obj.data.loop_triangles)
        except Exception:
            tris += len(obj.data.polygons) * 2
    return obj_count, mesh_count, verts, edges, tris


def has_normals_and_uvs() -> tuple[bool, bool]:
    require_blender()
    has_normals = True
    has_uvs = True
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        if not mesh.vertices or all(
            (v.normal.x == 0 and v.normal.y == 0 and v.normal.z == 0)
            for v in mesh.vertices
        ):
            has_normals = False
        if not mesh.uv_layers:
            has_uvs = False
    return has_normals, has_uvs


def count_materials() -> int:
    require_blender()
    return len({slot.material for obj in bpy.data.objects if obj.type == "MESH"
                for slot in obj.material_slots if slot.material})


def run_stage2() -> AnalysisResult:
    """Build the Stage-2 analysis report for the current scene."""
    require_blender()
    LOG.info("Stage 2 — analysing scene.")

    points = combined_world_points()
    # Convert metres → mm for reporting (consistent with deformation engine).
    points_mm = points * 1000.0 if points.size else points

    bbox_mm = compute_bbox(points_mm)
    size = bbox_mm.size

    obj_count, mesh_count, verts, edges, tris = count_topology()
    normals_ok, uvs_ok = has_normals_and_uvs()
    components = count_connected_components(points_mm, voxel=0.5)
    sym = symmetry_score(points_mm, axis="x")

    result = AnalysisResult(
        bbox_mm=bbox_mm,
        frame_width_mm=float(size[0]),
        frame_height_mm=float(size[1]),
        frame_depth_mm=float(size[2]),
        triangle_count=int(tris),
        vertex_count=int(verts),
        material_count=count_materials(),
        object_count=obj_count,
        mesh_count=mesh_count,
        has_normals=normals_ok,
        has_uvs=uvs_ok,
        connected_components=int(components),
        symmetry_score=float(sym),
        raw={
            "edge_count": int(edges),
        },
    )

    LOG.info(
        "Stage 2 — bbox %.1f×%.1f×%.1f mm, verts=%d tris=%d meshes=%d components=%d sym=%.3f",
        result.frame_width_mm,
        result.frame_height_mm,
        result.frame_depth_mm,
        result.vertex_count,
        result.triangle_count,
        result.mesh_count,
        result.connected_components,
        result.symmetry_score,
    )

    return result


def to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Serialise the analysis to a plain dict for logging / metadata."""
    payload = asdict(result)
    payload["bbox_mm"] = result.bbox_mm.to_dict()
    return payload
