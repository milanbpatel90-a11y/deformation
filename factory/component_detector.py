"""Stage 3 — Component detection.

Classifies every mesh (or sub-mesh island) of the imported scene as one of:

* Frame
* LeftLens / RightLens
* LeftRim / RightRim
* Bridge
* LeftTemple / RightTemple
* TempleTips
* NosePads

The detector combines six independent signals:

1. **Name heuristics** — keywords in the Blender object name.
2. **Material slots** — material names that include hints like "lens" or
   "temple".
3. **Bounding-box heuristics** — width, aspect ratio, and position in the
   XZ-plane.
4. **Connected components** — voxel-hashed islands of the combined point
   cloud, used when everything is merged into one mesh.
5. **Symmetry axis** — the cluster to the left of the YZ-plane is the
   *left* variant; the cluster to the right is *right*.
6. **Topology graph** — connected mesh islands within a single object,
   detected via bmesh traversal.

The classifier is intentionally permissive: every classification carries a
``confidence`` score in ``[0, 1]``. Down-stream stages are free to discard
low-confidence assignments or to apply additional topology cues.
"""

from __future__ import annotations

import re
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

from .types import BoundingBox, ClassifiedPart, ComponentClassification, PartKind
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------

_NAME_KEYWORDS: dict[PartKind, tuple[str, ...]] = {
    PartKind.LEFT_LENS: ("left_lens", "leftlens", "l_lens", "lens_l", "lens_left"),
    PartKind.RIGHT_LENS: ("right_lens", "rightlens", "r_lens", "lens_r", "lens_right"),
    PartKind.LEFT_RIM: ("left_rim", "leftrim", "rim_left", "rim_l"),
    PartKind.RIGHT_RIM: ("right_rim", "rightrim", "rim_right", "rim_r"),
    PartKind.BRIDGE: ("bridge", "nose_bridge", "nosepiece"),
    PartKind.LEFT_TEMPLE: ("left_temple", "lefttemple", "temple_left", "temple_l", "arm_left"),
    PartKind.RIGHT_TEMPLE: ("right_temple", "righttemple", "temple_right", "temple_r", "arm_right"),
    PartKind.TEMPLE_TIPS: ("temple_tip", "temple_end", "earpiece", "tip"),
    PartKind.NOSE_PADS: ("nose_pad", "nosepad", "nose_pad_left", "nose_pad_right", "pad"),
    PartKind.FRAME: ("frame", "front", "rim_full", "full_frame"),
}

_MATERIAL_KEYWORDS: dict[PartKind, tuple[str, ...]] = {
    PartKind.LEFT_LENS: ("left_lens", "lens_l", "lens_left"),
    PartKind.RIGHT_LENS: ("right_lens", "lens_r", "lens_right"),
    PartKind.LEFT_RIM: ("left_rim", "rim_l", "rim_left"),
    PartKind.RIGHT_RIM: ("right_rim", "rim_r", "rim_right"),
    PartKind.BRIDGE: ("bridge", "nose_bridge"),
    PartKind.LEFT_TEMPLE: ("left_temple", "temple_l", "arm_left"),
    PartKind.RIGHT_TEMPLE: ("right_temple", "temple_r", "arm_right"),
    PartKind.NOSE_PADS: ("nose_pad", "nosepad"),
    PartKind.FRAME: ("frame", "front"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.component_detector requires Blender.")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _match_keywords(text: str, bank: dict[PartKind, tuple[str, ...]]) -> list[tuple[PartKind, float]]:
    norm = _normalise(text)
    matches: list[tuple[PartKind, float]] = []
    for kind, keywords in bank.items():
        for kw in keywords:
            kw_norm = _normalise(kw)
            if not kw_norm:
                continue
            if kw_norm in norm:
                matches.append((kind, 0.95))
                break
            if norm.startswith(kw_norm) or norm.endswith(kw_norm):
                matches.append((kind, 0.7))
                break
    return matches


def classify_by_name(obj_name: str) -> list[tuple[PartKind, float]]:
    return _match_keywords(obj_name, _NAME_KEYWORDS)


def classify_by_material(material_names: Iterable[str]) -> list[tuple[PartKind, float]]:
    matches: list[tuple[PartKind, float]] = []
    for mat in material_names:
        matches.extend(_match_keywords(mat, _MATERIAL_KEYWORDS))
    return matches


@dataclass(slots=True)
class _Island:
    """A connected mesh island inside a single Blender object."""

    object_name: str
    object_ref: Any
    vertex_indices: np.ndarray
    center: np.ndarray
    bbox: BoundingBox
    is_separated_object: bool


def _collect_islands() -> list[_Island]:
    """Collect every connected mesh island in the scene.

    If a Blender object is its own island (``obj.data`` has a single connected
    component) we still emit it as an island but flag ``is_separated_object``
    so the splitter can skip a redundant ``separate``.
    """
    require_blender()
    islands: list[_Island] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            eval_obj.to_mesh_clear()
            continue

        # Build a bmesh and find connected components by face walks.
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        seen: set[int] = set()
        components: list[set[int]] = []
        for face in bm.faces:
            if face.index in seen:
                continue
            stack = [face]
            comp: set[int] = set()
            while stack:
                f = stack.pop()
                if f.index in seen:
                    continue
                seen.add(f.index)
                comp.add(f.index)
                for edge in f.edges:
                    for linked in edge.link_faces:
                        if linked.index not in seen:
                            stack.append(linked)
            components.append(comp)

        mw = obj.matrix_world
        for comp in components:
            face_idxs = sorted(comp)
            vert_set: set[int] = set()
            for fi in face_idxs:
                face = bm.faces[fi]
                for v in face.verts:
                    vert_set.add(v.index)
            vert_idxs = np.array(sorted(vert_set), dtype=np.int32)
            if vert_idxs.size == 0:
                continue
            world_coords = np.array(
                [(mw @ mesh.vertices[vi].co)[:] for vi in vert_idxs],
                dtype=np.float64,
            )
            mn = world_coords.min(axis=0)
            mx = world_coords.max(axis=0)
            center = world_coords.mean(axis=0)
            bbox = BoundingBox(
                min=(float(mn[0]), float(mn[1]), float(mn[2])),
                max=(float(mx[0]), float(mx[1]), float(mx[2])),
            )
            islands.append(
                _Island(
                    object_name=obj.name,
                    object_ref=obj,
                    vertex_indices=vert_idxs,
                    center=center,
                    bbox=bbox,
                    is_separated_object=(len(components) == 1),
                )
            )
        bm.free()
        eval_obj.to_mesh_clear()

    return islands


def _island_bbox_size_mm(island: _Island) -> tuple[float, float, float]:
    s = island.bbox.size
    return s[0] * 1000.0, s[1] * 1000.0, s[2] * 1000.0


def _classify_by_geometry(island: _Island, frame_bbox: BoundingBox) -> tuple[PartKind, float]:
    """Use bbox shape and position to assign a kind.

    Assumes the model is in canonical orientation (temples along ±X,
    lenses in front). All distances in mm.
    """
    w_mm, h_mm, d_mm = _island_bbox_size_mm(island)
    frame_w = frame_bbox.size[0] * 1000.0
    frame_h = frame_bbox.size[1] * 1000.0
    cx = island.center[0] * 1000.0
    cy = island.center[1] * 1000.0

    # ---- Temple heuristic ------------------------------------------------
    # A temple is long, thin, sits at the extreme +X or -X of the bbox,
    # and spans most of the Y-extent of the frame.
    if (
        w_mm > 0.6 * frame_w
        and w_mm > 2.0 * h_mm
        and (cx > 0.4 * frame_w or cx < -0.4 * frame_w)
        and d_mm < h_mm * 1.2
    ):
        return (PartKind.LEFT_TEMPLE if cx < 0 else PartKind.RIGHT_TEMPLE), 0.85

    # ---- Lens/rim heuristic ---------------------------------------------
    # A lens or rim island is roughly the size of half the frame width and
    # sits in the front-half of the depth.
    if (
        0.2 * frame_w < w_mm < 0.55 * frame_w
        and 0.2 * frame_h < h_mm < 0.9 * frame_h
        and (cx > 0 or cx < 0)
        and abs(cx) > 0.15 * frame_w
    ):
        if d_mm < 0.35 * frame_w:
            return (PartKind.LEFT_LENS if cx < 0 else PartKind.RIGHT_LENS), 0.55
        return (PartKind.LEFT_RIM if cx < 0 else PartKind.RIGHT_RIM), 0.65

    # ---- Bridge heuristic -----------------------------------------------
    # A bridge is centred on X, sits at the top half of Y, and is small in X
    # but sizeable in Y.
    if abs(cx) < 0.15 * frame_w and 0.15 * frame_h < h_mm < 0.6 * frame_h and w_mm < 0.25 * frame_w:
        return PartKind.BRIDGE, 0.6

    # ---- Nose pads heuristic --------------------------------------------
    # Pads are small, sit just below the bridge (cy < 0), and on either side
    # of X.
    if (
        abs(cx) > 0.05 * frame_w
        and cy < 0
        and w_mm < 0.15 * frame_w
        and h_mm < 0.2 * frame_h
    ):
        return (PartKind.LEFT_TEMPLE if cx < 0 else PartKind.RIGHT_TEMPLE), 0.3  # could also be a pad

    return PartKind.UNKNOWN, 0.0


def _classify_single(island: _Island, frame_bbox: BoundingBox) -> ClassifiedPart:
    """Classify a single island into a ClassifiedPart."""
    name_matches = classify_by_name(island.object_name)
    material_names: list[str] = []
    obj = island.object_ref
    if obj is not None:
        for slot in obj.material_slots:
            if slot.material and slot.material.name:
                material_names.append(slot.material.name)
    mat_matches = classify_by_material(material_names)

    geo_kind, geo_conf = _classify_by_geometry(island, frame_bbox)

    # Aggregate evidence.
    candidates: dict[PartKind, float] = {}
    for kind, conf in name_matches:
        candidates[kind] = max(candidates.get(kind, 0.0), conf)
    for kind, conf in mat_matches:
        candidates[kind] = max(candidates.get(kind, 0.0), conf * 0.9)
    if geo_kind != PartKind.UNKNOWN:
        candidates[geo_kind] = max(candidates.get(geo_kind, 0.0), geo_conf)

    # If the dominant signal is lens-vs-rim ambiguity, prefer the rim unless
    # the island is extremely thin in depth.
    if (
        PartKind.LEFT_LENS in candidates
        and PartKind.LEFT_RIM in candidates
    ):
        candidates.pop(PartKind.LEFT_LENS)
    if (
        PartKind.RIGHT_LENS in candidates
        and PartKind.RIGHT_RIM in candidates
    ):
        candidates.pop(PartKind.RIGHT_LENS)

    if not candidates:
        # Default: anything that big and central is the frame.
        w_mm, h_mm, _ = _island_bbox_size_mm(island)
        if (
            w_mm > 0.55 * frame_bbox.size[0] * 1000.0
            and h_mm > 0.45 * frame_bbox.size[1] * 1000.0
        ):
            kind = PartKind.FRAME
            confidence = 0.5
        else:
            kind = PartKind.UNKNOWN
            confidence = 0.0
    else:
        kind, confidence = max(candidates.items(), key=lambda kv: kv[1])

    # Fallback: anything tagged as a "temple" by geometry but on the side
    # far from the hinge should be downgraded to temple_tip when very short.
    w_mm, h_mm, _ = _island_bbox_size_mm(island)
    if kind in (PartKind.LEFT_TEMPLE, PartKind.RIGHT_TEMPLE) and w_mm < 0.35 * frame_bbox.size[0] * 1000.0:
        kind = PartKind.TEMPLE_TIPS
        confidence = min(confidence, 0.5)

    return ClassifiedPart(
        name=island.object_name,
        kind=kind,
        vertex_count=int(island.vertex_indices.size),
        triangle_count=int(island.vertex_indices.size // 3),
        bbox=island.bbox,
        center=(float(island.center[0]), float(island.center[1]), float(island.center[2])),
        confidence=float(confidence),
        source_object=island.object_name,
        is_separated_mesh=island.is_separated_object,
        material=material_names[0] if material_names else None,
    )


def _ensure_frame_assignment(parts: list[ClassifiedPart], frame_bbox: BoundingBox) -> None:
    """If no part is tagged ``FRAME``, promote the largest central island."""
    if any(p.kind == PartKind.FRAME for p in parts):
        return
    if not parts:
        return
    candidates = sorted(parts, key=lambda p: -p.bbox.size[0] * p.bbox.size[1] * p.bbox.size[2])
    for cand in candidates:
        if cand.kind in (PartKind.UNKNOWN, PartKind.FRAME):
            cand.kind = PartKind.FRAME
            cand.confidence = max(cand.confidence, 0.4)
            LOG.info("Promoted %s → FRAME (fallback).", cand.name)
            return


def _balance_sides(parts: list[ClassifiedPart]) -> None:
    """Ensure left/right symmetry of paired parts (lens, rim, temple).

    If the classifier identified only a left part, mirror its confidence to
    its right counterpart on the opposite side of X, and vice versa.
    """
    pair_map: dict[tuple[PartKind, PartKind], str] = {
        (PartKind.LEFT_LENS, PartKind.RIGHT_LENS): "x",
        (PartKind.LEFT_RIM, PartKind.RIGHT_RIM): "x",
        (PartKind.LEFT_TEMPLE, PartKind.RIGHT_TEMPLE): "x",
    }
    for left_kind, right_kind in pair_map:
        left = next((p for p in parts if p.kind == left_kind), None)
        right = next((p for p in parts if p.kind == right_kind), None)
        if left and not right:
            clone = ClassifiedPart(
                name=f"{left.name}_mirror",
                kind=right_kind,
                vertex_count=left.vertex_count,
                triangle_count=left.triangle_count,
                bbox=left.bbox,
                center=(-left.center[0], left.center[1], left.center[2]),
                confidence=left.confidence * 0.6,
                source_object=left.source_object,
                is_separated_mesh=False,
                material=left.material,
            )
            parts.append(clone)
        elif right and not left:
            clone = ClassifiedPart(
                name=f"{right.name}_mirror",
                kind=left_kind,
                vertex_count=right.vertex_count,
                triangle_count=right.triangle_count,
                bbox=right.bbox,
                center=(-right.center[0], right.center[1], right.center[2]),
                confidence=right.confidence * 0.6,
                source_object=right.source_object,
                is_separated_mesh=False,
                material=right.material,
            )
            parts.append(clone)


def run_stage3(analysis_bbox: BoundingBox) -> ComponentClassification:
    """Run the Stage-3 classification pass on the current scene."""
    require_blender()
    LOG.info("Stage 3 — detecting components.")

    islands = _collect_islands()
    if not islands:
        raise RuntimeError("No mesh islands found; cannot classify components.")

    parts = [_classify_single(island, analysis_bbox) for island in islands]

    # Multiple islands of the same source object (a merged mesh) need to be
    # de-duplicated by keeping the best kind per kind+side. We collapse by
    # (kind, source_object), preferring the highest confidence.
    collapsed: dict[tuple[PartKind, str], ClassifiedPart] = {}
    for part in parts:
        key = (part.kind, part.source_object)
        if key not in collapsed or collapsed[key].confidence < part.confidence:
            collapsed[key] = part

    deduped = list(collapsed.values())

    _ensure_frame_assignment(deduped, analysis_bbox)
    _balance_sides(deduped)

    classification = ComponentClassification(parts=deduped, unknown_objects=[])
    for part in deduped:
        if part.kind == PartKind.FRAME:
            classification.frame_part = part.name
        elif part.kind == PartKind.LEFT_LENS:
            classification.left_lens_part = part.name
        elif part.kind == PartKind.RIGHT_LENS:
            classification.right_lens_part = part.name
        elif part.kind == PartKind.BRIDGE:
            classification.bridge_part = part.name
        elif part.kind == PartKind.LEFT_TEMPLE:
            classification.left_temple_part = part.name
        elif part.kind == PartKind.RIGHT_TEMPLE:
            classification.right_temple_part = part.name

    for part in deduped:
        LOG.info(
            "Classified %-20s → %-12s conf=%.2f verts=%d",
            part.name,
            part.kind.value,
            part.confidence,
            part.vertex_count,
        )

    return classification
