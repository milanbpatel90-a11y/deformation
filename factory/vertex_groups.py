"""Stage 5 — Topological vertex groups.

Creates the canonical ``VG_*`` vertex groups used by the deformation
engine. The required groups are::

    VG_FRAME         — every vertex of the Frame object
    VG_LEFT_RIM      — left half of the Frame (border-loop region adjacent to LeftLens)
    VG_RIGHT_RIM     — right half of the Frame
    VG_LEFT_LENS     — every vertex of the LeftLens object
    VG_RIGHT_LENS    — every vertex of the RightLens object
    VG_BRIDGE        — every vertex of the Bridge object
    VG_LEFT_TEMPLE   — every vertex of the LeftTemple object
    VG_RIGHT_TEMPLE  — every vertex of the RightTemple object

For the Frame object we *also* subdivide by border-loop analysis rather
than relying on the bounding box: vertices on the left half of the
topology graph (closest to the LeftLens centroid) belong to
``VG_LEFT_RIM``; the rest belong to ``VG_RIGHT_RIM``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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

from .types import ClassifiedPart, ComponentClassification, PartKind, VertexGroupSpec
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Group name constants
# ---------------------------------------------------------------------------

VG_FRAME = "VG_FRAME"
VG_LEFT_RIM = "VG_LEFT_RIM"
VG_RIGHT_RIM = "VG_RIGHT_RIM"
VG_LEFT_LENS = "VG_LEFT_LENS"
VG_RIGHT_LENS = "VG_RIGHT_LENS"
VG_BRIDGE = "VG_BRIDGE"
VG_LEFT_TEMPLE = "VG_LEFT_TEMPLE"
VG_RIGHT_TEMPLE = "VG_RIGHT_TEMPLE"

CANONICAL_OBJECT_TO_GROUP: dict[str, str] = {
    "Frame": VG_FRAME,
    "LeftRim": VG_LEFT_RIM,
    "RightRim": VG_RIGHT_RIM,
    "LeftLens": VG_LEFT_LENS,
    "RightLens": VG_RIGHT_LENS,
    "Bridge": VG_BRIDGE,
    "LeftTemple": VG_LEFT_TEMPLE,
    "RightTemple": VG_RIGHT_TEMPLE,
    "TempleTips": "VG_TEMPLE_TIPS",
    "NosePads": "VG_NOSE_PADS",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.vertex_groups requires Blender.")


@dataclass(slots=True)
class VertexGroupBuildResult:
    """Outcome of the vertex-group pass."""

    specs: list[VertexGroupSpec]
    per_object_groups: dict[str, list[str]]


def _ensure_group(obj, group_name: str) -> "bpy.types.VertexGroup":
    require_blender()
    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        vg = obj.vertex_groups.new(name=group_name)
    return vg


def _clear_group(vg) -> None:
    """Remove every assignment from the given vertex group."""
    try:
        # ``remove`` with ``all=True`` is supported in 3.x.
        vg.remove(list(range(_vertex_count(vg.id_data))))
    except (TypeError, RuntimeError):
        # Fallback: remove one vertex at a time.
        for v in vg.id_data.data.vertices:
            vg.remove([v.index])


def _vertex_count(mesh) -> int:
    return len(mesh.vertices)


def _add_vertices_to_group(vg, indices: Iterable[int], weight: float = 1.0) -> None:
    obj = vg.id_data
    obj_data = obj.data
    # ``add`` accepts a list of (index, weight, type) tuples when given a
    # list of vertex indices plus an explicit weight argument, but the
    # safest cross-version path is to call add() per vertex.
    for idx in indices:
        try:
            vg.add([int(idx)], float(weight), "REPLACE")
        except TypeError:
            # Older API: add(index, weight, type) — deprecated but works.
            vg.add(int(idx), float(weight), "REPLACE")


# ---------------------------------------------------------------------------
# Object-level groups (one object → one group containing all its vertices)
# ---------------------------------------------------------------------------

def _build_object_group(obj, group_name: str) -> VertexGroupSpec:
    require_blender()
    vg = _ensure_group(obj, group_name)
    _clear_group(vg)
    verts = list(range(len(obj.data.vertices)))
    _add_vertices_to_group(vg, verts, weight=1.0)
    return VertexGroupSpec(name=group_name, indices=list(verts))


# ---------------------------------------------------------------------------
# Topology-based frame subdivision
# ---------------------------------------------------------------------------

def _build_frame_subgroups(frame_obj, left_anchor: np.ndarray | None, right_anchor: np.ndarray | None) -> tuple[VertexGroupSpec, VertexGroupSpec]:
    """Split a Frame mesh into LEFT_RIM / RIGHT_RIM by graph propagation.

    The algorithm:

    1. Identify two seed vertices — one closest to ``left_anchor`` and one
       closest to ``right_anchor``.
    2. Flood-fill the vertex graph outward, assigning each visited vertex
       to the closest seed in edge-graph distance.
    3. Vertices equidistant from both seeds fall to the side their centroid
       is closer to.
    """
    require_blender()
    mesh = frame_obj.data
    n = len(mesh.vertices)
    if n == 0:
        empty = VertexGroupSpec(name=VG_LEFT_RIM, indices=[])
        empty_other = VertexGroupSpec(name=VG_RIGHT_RIM, indices=[])
        return empty, empty_other

    coords = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for edge in mesh.edges:
        a, b = edge.vertices[0], edge.vertices[1]
        if a == b:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    if left_anchor is None or right_anchor is None:
        # Fall back to bbox-only splitting.
        mid_x = float(np.mean(coords[:, 0]))
        labels = np.where(coords[:, 0] <= mid_x, 0, 1)
    else:
        # Seed selection.
        d_left = np.linalg.norm(coords - left_anchor, axis=1)
        d_right = np.linalg.norm(coords - right_anchor, axis=1)
        seed_left = int(np.argmin(d_left))
        seed_right = int(np.argmin(d_right))

        # BFS from both seeds in parallel. Track (best_side, distance).
        best = np.full(n, -1, dtype=np.int8)
        dist = np.full(n, np.inf, dtype=np.float64)
        queue: deque[tuple[int, int]] = deque()
        queue.append((seed_left, 0))
        queue.append((seed_right, 1))
        best[seed_left] = 0
        best[seed_right] = 1
        dist[seed_left] = 0.0
        dist[seed_right] = 0.0

        while queue:
            v, side = queue.popleft()
            d = dist[v] + 1.0
            for nb in adjacency[v]:
                if d < dist[nb]:
                    dist[nb] = d
                    best[nb] = side
                    queue.append((nb, side))
                elif d == dist[nb] and best[nb] != side:
                    # Tie-break: assign to the side whose seed is closer
                    # in 3D space.
                    best[nb] = 0 if d_left[nb] < d_right[nb] else 1
        labels = best

    left_idx = [int(i) for i in np.where(labels == 0)[0]]
    right_idx = [int(i) for i in np.where(labels == 1)[0]]

    return (
        VertexGroupSpec(name=VG_LEFT_RIM, indices=left_idx),
        VertexGroupSpec(name=VG_RIGHT_RIM, indices=right_idx),
    )


def _world_anchor(part: ClassifiedPart | None) -> np.ndarray | None:
    if part is None:
        return None
    return np.array(part.center, dtype=np.float64)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_stage5(classification: ComponentClassification) -> VertexGroupBuildResult:
    require_blender()
    LOG.info("Stage 5 — building topological vertex groups.")

    frame_obj = bpy.data.objects.get(classification.frame_part or "Frame")
    left_rim_part = next((p for p in classification.parts if p.kind == PartKind.LEFT_RIM), None)
    right_rim_part = next((p for p in classification.parts if p.kind == PartKind.RIGHT_RIM), None)
    left_lens_part = next((p for p in classification.parts if p.kind == PartKind.LEFT_LENS), None)
    right_lens_part = next((p for p in classification.parts if p.kind == PartKind.RIGHT_LENS), None)

    specs: list[VertexGroupSpec] = []
    per_object: dict[str, list[str]] = {}

    # Whole-object groups.
    for canonical_name, group_name in CANONICAL_OBJECT_TO_GROUP.items():
        obj = bpy.data.objects.get(canonical_name)
        if obj is None:
            continue
        spec = _build_object_group(obj, group_name)
        specs.append(spec)
        per_object.setdefault(canonical_name, []).append(group_name)

    # Topology-based split of the Frame.
    if frame_obj is not None:
        left_anchor = _world_anchor(left_rim_part) or _world_anchor(left_lens_part)
        right_anchor = _world_anchor(right_rim_part) or _world_anchor(right_lens_part)
        left_spec, right_spec = _build_frame_subgroups(frame_obj, left_anchor, right_anchor)

        left_vg = _ensure_group(frame_obj, left_spec.name)
        _clear_group(left_vg)
        _add_vertices_to_group(left_vg, left_spec.indices, weight=1.0)

        right_vg = _ensure_group(frame_obj, right_spec.name)
        _clear_group(right_vg)
        _add_vertices_to_group(right_vg, right_spec.indices, weight=1.0)

        # Override the Frame's whole-mesh VG_FRAME with the actual subset of
        # vertices that participate in the rim subdivision (a sanity-check
        # invariant: every frame vertex belongs to exactly one of the two rims).
        all_rim = set(left_spec.indices) | set(right_spec.indices)
        frame_vg = _ensure_group(frame_obj, VG_FRAME)
        _clear_group(frame_vg)
        _add_vertices_to_group(frame_vg, sorted(all_rim), weight=1.0)

        specs.append(left_spec)
        specs.append(right_spec)
        per_object.setdefault("Frame", []).extend([VG_FRAME, VG_LEFT_RIM, VG_RIGHT_RIM])

    LOG.info(
        "Stage 5 done — %d vertex groups across %d objects.",
        sum(len(v) for v in per_object.values()),
        len(per_object),
    )

    return VertexGroupBuildResult(specs=specs, per_object_groups=per_object)
