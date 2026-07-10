"""Stage 1 — Import, clean, and normalise raw GLB models.

This module must run inside Blender's Python runtime (``bpy`` is unavailable
otherwise). When imported outside Blender, all functions raise ``ImportError``
early so the rest of the pipeline is never silently corrupted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import bpy  # type: ignore
    import mathutils  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - non-Blender host
    bpy = None  # type: ignore[assignment]
    mathutils = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False
    _BLENDER_IMPORT_ERROR = _exc

from .utils import TARGET_FRAME_WIDTH_MM, get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ImportResult:
    """Outcome of importing a single raw GLB."""

    source_path: Path
    selected_objects: list[str]
    bbox_world_min: tuple[float, float, float]
    bbox_world_max: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    detected_unit_scale: float     # factor to multiply metres→current unit
    applied_scale: float           # final scale applied to reach 140 mm width
    object_count: int
    mesh_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    """Raise if not running inside Blender."""
    if not _BLENDER_AVAILABLE:
        raise ImportError(
            "factory.importer requires Blender's Python runtime. "
            f"Original error: {_BLENDER_IMPORT_ERROR!r}"
        )


def reset_scene() -> None:
    """Wipe the current Blender scene to a clean state."""
    require_blender()
    # Delete every object. This also strips default cube / camera / light.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan data blocks so reused names do not collide.
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.curves,
    ):
        for block in list(collection):
            collection.remove(block)


def import_glb(path: Path) -> None:
    """Import a GLB file and parent everything to the active collection."""
    require_blender()
    if not path.exists():
        raise FileNotFoundError(f"GLB not found: {path}")

    before_objects = set(_all_object_names())
    # ``files=[...]`` is the documented way to drive the operator from Python.
    bpy.ops.import_scene.gltf(
        filepath=str(path),
        files=[{"name": path.name}],
        loglevel=0,
        import_pack_images=True,
        merge_vertices=True,
        import_shading="NORMAL",
        bone_heuristic="BLENDER",
    )
    after_objects = set(_all_object_names())
    new_objects = after_objects - before_objects
    if not new_objects:
        raise RuntimeError(f"No new objects were imported from {path}")
    LOG.info("Imported %d objects from %s", len(new_objects), path.name)


def delete_non_mesh_objects(keep: Iterable[str] | None = None) -> int:
    """Delete all cameras, lights, empties, etc. Keep only mesh objects.

    If ``keep`` is provided, those object names are preserved even if they
    are not meshes.
    """

    require_blender()
    keep_set = set(keep or ())
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.name in keep_set:
            continue
        if obj.type == "MESH":
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
        removed += 1
    return removed


def _all_object_names() -> list[str]:
    return [obj.name for obj in bpy.data.objects]


def collect_mesh_objects() -> list:
    """Return every mesh object in the scene."""
    require_blender()
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def apply_transforms(rotate: bool = True, scale: bool = True) -> None:
    """Apply rotation/scale so the mesh data carries the world-space transform."""
    require_blender()
    if not collect_mesh_objects():
        return
    bpy.ops.object.select_all(action="DESELECT")
    for obj in collect_mesh_objects():
        obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.view_layer.objects.active or collect_mesh_objects()[0]
    bpy.ops.object.transform_apply(
        location=False,
        rotation=rotate,
        scale=scale,
    )


def _scene_bbox_world(objects) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute combined world-space bounding box of the given mesh objects."""
    require_blender()
    min_v = [math.inf, math.inf, math.inf]
    max_v = [-math.inf, -math.inf, -math.inf]
    found = False
    for obj in objects:
        if obj.type != "MESH":
            continue
        # Force evaluation to refresh evaluated depsgraph.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            continue
        found = True
        # Transform vertices into world space manually for accuracy.
        mw = obj.matrix_world
        for v in mesh.vertices:
            wp = mw @ mathutils.Vector((v.co.x, v.co.y, v.co.z))
            for i in range(3):
                if wp[i] < min_v[i]:
                    min_v[i] = wp[i]
                if wp[i] > max_v[i]:
                    max_v[i] = wp[i]
        eval_obj.to_mesh_clear()
    if not found:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return tuple(min_v), tuple(max_v)  # type: ignore[return-value]


def detect_unit_scale(bbox_size: tuple[float, float, float]) -> float:
    """Guess the multiplicative factor to convert ``bbox_size`` to metres.

    Blender's default unit is the metre. GLB files written in centimetres will
    report widths of 13.5 instead of 0.135; we therefore probe the largest
    extent. Returns 1.0 (already metres) for plausible eyewear widths in
    0.05–0.5 m, 100 for centimetre-scale, 1000 for millimetre-scale.
    """

    width = max(bbox_size)
    if width <= 0.0:
        return 1.0
    if width < 0.05:
        # Already millimetres-accurate, treat as metres.
        return 1.0
    if width < 0.5:
        return 1.0
    if width < 50.0:
        return 0.01  # input is centimetres
    return 0.001  # input is millimetres


def center_at_origin(objects) -> None:
    """Translate meshes so their combined centre is at the world origin."""
    require_blender()
    bbox_min, bbox_max = _scene_bbox_world(objects)
    if not any(math.isfinite(v) for v in bbox_min + bbox_max):
        return
    cx = 0.5 * (bbox_min[0] + bbox_max[0])
    cy = 0.5 * (bbox_min[1] + bbox_max[1])
    cz = 0.5 * (bbox_min[2] + bbox_max[2])
    shift = mathutils.Vector((-cx, -cy, -cz))
    if shift.length < 1e-9:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None
    bpy.ops.transform.translate(value=shift)


def normalize_width(
    objects,
    *,
    target_mm: float = TARGET_FRAME_WIDTH_MM,
    axis: str = "x",
) -> float:
    """Scale meshes so the bbox width along ``axis`` equals ``target_mm``.

    Returns the multiplier applied (BEFORE the post-scaling transform apply).
    """
    require_blender()
    bbox_min, bbox_max = _scene_bbox_world(objects)
    width = bbox_max["xyz".index(axis)] - bbox_min["xyz".index(axis)]
    if width <= 1e-9:
        LOG.warning("Cannot normalise: bbox width is degenerate (%.6f).", width)
        return 1.0
    # 1 Blender unit = 1 m. Convert target metres → Blender units, divide by
    # current width to get the scale factor.
    target_units = target_mm / 1000.0
    factor = target_units / width
    if not (0.001 < factor < 1000.0):
        LOG.warning("Scale factor %.3f is outside the safe window.", factor)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None
    bpy.ops.transform.resize(value=(factor, factor, factor))
    return factor


def set_origin_to_geometry(objects) -> None:
    """Move each mesh's origin to its geometry centre (object-level)."""
    require_blender()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.type == "MESH":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
            obj.select_set(False)


def run_stage1(
    glb_path: Path,
    *,
    target_width_mm: float = TARGET_FRAME_WIDTH_MM,
) -> ImportResult:
    """Execute the full Stage-1 pipeline on a single GLB.

    Returns the geometric summary used by Stage 2. Re-raises on failure so
    the caller can decide whether to abort the batch.
    """

    require_blender()
    LOG.info("Stage 1 — importing %s", glb_path.name)
    reset_scene()
    import_glb(glb_path)

    removed = delete_non_mesh_objects()
    LOG.info("Deleted %d non-mesh objects (cameras/lights/etc.).", removed)

    meshes = collect_mesh_objects()
    if not meshes:
        raise RuntimeError(f"No mesh objects after import of {glb_path}")

    # First pass: detect unit, scale if needed, then centre.
    bbox_min, bbox_max = _scene_bbox_world(meshes)
    bbox_size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
    unit_scale = detect_unit_scale(bbox_size)

    if abs(unit_scale - 1.0) > 1e-6:
        LOG.info(
            "Detected non-metre units (%.4f) — pre-scaling meshes.", unit_scale
        )
        # Multiply each object's scale to convert to metres, then apply.
        for obj in meshes:
            obj.scale = (
                obj.scale[0] * unit_scale,
                obj.scale[1] * unit_scale,
                obj.scale[2] * unit_scale,
            )
        apply_transforms(rotate=False, scale=True)
        meshes = collect_mesh_objects()
        bbox_min, bbox_max = _scene_bbox_world(meshes)
        bbox_size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))

    # Rotate so the glasses face +Z and temples extend along ±X.
    _orient_glasses_to_canonical(meshes)

    apply_transforms(rotate=True, scale=True)
    set_origin_to_geometry(meshes)

    center_at_origin(meshes)
    apply_transforms(rotate=False, scale=False)

    bbox_min, bbox_max = _scene_bbox_world(meshes)
    bbox_size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))

    applied_scale = normalize_width(meshes, target_mm=target_width_mm, axis="x")
    apply_transforms(rotate=True, scale=True)
    set_origin_to_geometry(meshes)
    center_at_origin(meshes)
    apply_transforms(rotate=False, scale=False)

    bbox_min, bbox_max = _scene_bbox_world(meshes)
    bbox_size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))

    meshes = collect_mesh_objects()
    LOG.info(
        "Stage 1 done — %d meshes, bbox=%s",
        len(meshes),
        _fmt_size(bbox_size),
    )

    return ImportResult(
        source_path=glb_path,
        selected_objects=[o.name for o in meshes],
        bbox_world_min=tuple(bbox_min),
        bbox_world_max=tuple(bbox_max),
        bbox_size=bbox_size,
        detected_unit_scale=unit_scale,
        applied_scale=applied_scale,
        object_count=len(bpy.data.objects),
        mesh_count=len(meshes),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _orient_glasses_to_canonical(meshes) -> None:
    """Rotate the meshes so the principal axes match the canonical frame.

    Canonical convention:
      X — left↔right (temples extend along ±X)
      Y — vertical
      Z — depth (faces the camera)
    """
    require_blender()
    if not meshes:
        return
    # Compute the principal axes from the combined point cloud.
    verts = _combined_world_vertices(meshes)
    if verts.shape[0] < 3:
        return
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    # Use SVD on the covariance to find dominant axes.
    _, _, vh = np.linalg_svd_3x3(centered)
    # We want the world "up" axis to align with our +Y. If the source uses
    # +Y or +Z as up, rotate to bring up onto +Y.
    rows = vh
    x_axis, y_axis, z_axis = rows[0], rows[1], rows[2]
    # Decide which axis is vertical (largest absolute spread).
    spreads = [
        (float(np_ptp(centered @ x_axis)), x_axis, 0),
        (float(np_ptp(centered @ y_axis)), y_axis, 1),
        (float(np_ptp(centered @ z_axis)), z_axis, 2),
    ]
    spreads.sort(reverse=True)
    vertical_axis = spreads[0][1]
    horizontal_axis = spreads[1][1]
    depth_axis = spreads[2][1]
    # Build a rotation matrix that maps world axes onto canonical.
    target_up = np.array((0.0, 1.0, 0.0))
    rot = _rotation_aligning_pair(vertical_axis, target_up)
    if horizontal_axis is not None:
        # Make sure horizontal axis is roughly along X (positive).
        candidate = rot @ np.array((1.0, 0.0, 0.0))
        if float(candidate[0]) < 0.0:
            # Reflect along X (flip via rotation 180° around Y).
            flip = _y_flip_matrix()
            rot = flip @ rot
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    # Convert to Euler XYZ.
    eul = _matrix_to_euler_xyz(rot)
    bpy.ops.transform.rotate(value=eul[0], orient_axis="X")
    bpy.ops.transform.rotate(value=eul[1], orient_axis="Y")
    bpy.ops.transform.rotate(value=eul[2], orient_axis="Z")


def _combined_world_vertices(meshes) -> "numpy.ndarray":
    pieces = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            continue
        mw = obj.matrix_world
        coords = np.array(
            [(mw @ mathutils.Vector(v.co))[:] for v in mesh.vertices],
            dtype=np.float64,
        )
        pieces.append(coords)
        eval_obj.to_mesh_clear()
    if not pieces:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


class np:  # noqa: N801 - lightweight numpy stand-in for the two helpers we use
    @staticmethod
    def linalg_svd_3x3(matrix):  # type: ignore[no-untyped-def]
        import numpy as np_real
        return np_real.linalg.svd(matrix, full_matrices=False)

    @staticmethod
    def ptp(arr):  # type: ignore[no-untyped-def]
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        return float(arr.max() - arr.min())


import numpy as np  # noqa: E402  (real numpy re-import after shim)


def _rotation_aligning_pair(v_from: "np.ndarray", v_to: "np.ndarray") -> "np.ndarray":
    """Return a 3x3 rotation matrix that aligns ``v_from`` with ``v_to``."""
    v_from = np.asarray(v_from, dtype=np.float64)
    v_to = np.asarray(v_to, dtype=np.float64)
    a = v_from / max(np.linalg.norm(v_from), 1e-9)
    b = v_to / max(np.linalg.norm(v_to), 1e-9)
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cos_t > 0.99999:
        return np.eye(3)
    if cos_t < -0.99999:
        # 180° around any axis perpendicular to a.
        tmp = np.array((1.0, 0.0, 0.0)) if abs(a[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
        axis = np.cross(a, tmp)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        return _rotation_around_axis(axis, math.pi)
    axis = np.cross(a, b)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    angle = math.acos(cos_t)
    return _rotation_around_axis(axis, angle)


def _rotation_around_axis(axis: "np.ndarray", angle: float) -> "np.ndarray":
    a = np.asarray(axis, dtype=np.float64) / max(np.linalg.norm(axis), 1e-9)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _y_flip_matrix() -> "np.ndarray":
    # 180° rotation around Y.
    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def _matrix_to_euler_xyz(matrix: "np.ndarray") -> tuple[float, float, float]:
    """Extract Euler XYZ angles from a 3x3 rotation matrix (in radians)."""
    m = np.asarray(matrix, dtype=np.float64)
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(m[2, 1], m[2, 2])
        y = math.atan2(-m[2, 0], sy)
        z = math.atan2(m[1, 0], m[0, 0])
    else:
        x = math.atan2(-m[1, 2], m[1, 1])
        y = math.atan2(-m[2, 0], sy)
        z = 0.0
    return (x, y, z)


def _fmt_size(size: tuple[float, float, float]) -> str:
    # Convert metres → mm for readability.
    return f"{size[0] * 1000:.1f} × {size[1] * 1000:.1f} × {size[2] * 1000:.1f} mm"
