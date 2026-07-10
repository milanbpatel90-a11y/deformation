"""Stage 8 — Render thumbnails (front, side, 45°) at 1024×1024 PNG.

Uses Blender's Eevee or Workbench engine for speed. Background is transparent
(alpha channel = 1.0 for opaque parts, 0.0 for background).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import bpy  # type: ignore
    import mathutils  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    mathutils = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .types import ComponentClassification
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_blender() -> None:
    if not _BLENDER_AVAILABLE:
        raise ImportError("factory.thumbnail_renderer requires Blender.")


@dataclass(slots=True)
class ThumbnailResult:
    front: Path
    side: Path
    angle: Path
    resolution: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _configure_render_engine(resolution: int) -> None:
    require_blender()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "EEVEE_NEXT") else "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15

    # Eevee settings for clean orthographic look.
    eevee = scene.eevee
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 16
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = False
    if hasattr(eevee, "use_shadows"):
        eevee.use_shadows = True
    if hasattr(eevee, "shadow_method"):
        eevee.shadow_method = "VSM"
    # Use a simple world for even lighting.
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("FactoryWorld")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # white background for alpha
        bg.inputs[1].default_value = 1.5


def _ensure_camera() -> Any:
    require_blender()
    cam_data = bpy.data.cameras.new("FactoryCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.0
    cam_obj = bpy.data.objects.new("FactoryCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def _clear_lights() -> None:
    require_blender()
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def _add_three_point_lighting() -> None:
    require_blender()
    # Key light (45° front-top)
    key = bpy.data.lights.new("Key", "SUN")
    key.energy = 3.0
    key.angle = 0.1
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(-45), 0, math.radians(-30))
    bpy.context.collection.objects.link(key_obj)

    # Fill light (opposite side, softer)
    fill = bpy.data.lights.new("Fill", "SUN")
    fill.energy = 1.2
    fill.angle = 0.1
    fill_obj = bpy.data.objects.new("Fill", fill)
    fill_obj.rotation_euler = (math.radians(-30), 0, math.radians(150))
    bpy.context.collection.objects.link(fill_obj)

    # Rim light (from behind, highlights edges)
    rim = bpy.data.lights.new("Rim", "SUN")
    rim.energy = 1.5
    rim.angle = 0.1
    rim_obj = bpy.data.objects.new("Rim", rim)
    rim_obj.rotation_euler = (math.radians(20), 0, math.radians(180))
    bpy.context.collection.objects.link(rim_obj)


def _frame_camera_to_scene(cam_obj: Any, scale_factor: float = 1.15) -> None:
    """Adjust ortho_scale so the entire model fits with a small margin."""
    require_blender()
    # Collect all mesh vertices in world space.
    all_pts: list[np.ndarray] = []
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
        pts = np.array(
            [(mw @ v.co)[:] for v in mesh.vertices],
            dtype=np.float64,
        )
        all_pts.append(pts)
        eval_obj.to_mesh_clear()

    if not all_pts:
        return

    pts = np.concatenate(all_pts, axis=0)
    # Project onto camera plane (Z = camera forward direction)
    # Camera looks along -Z; we want X and Y extents in camera space.
    cam_matrix = cam_obj.matrix_world.inverted()
    cam_pts = pts @ cam_matrix.to_3x3().T + np.array(cam_matrix.translation)
    # cam_pts is in camera local space; camera looks along -Z.
    # So X is right/left, Y is up/down.
    min_x, max_x = float(cam_pts[:, 0].min()), float(cam_pts[:, 0].max())
    min_y, max_y = float(cam_pts[:, 1].min()), float(cam_pts[:, 1].max())

    width = max_x - min_x
    height = max_y - min_y
    ortho_scale = max(width, height) * scale_factor
    cam_obj.data.ortho_scale = max(ortho_scale, 1e-4)

    # Also center the camera on the model.
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    # Z position: far enough back to see everything.
    max_z = float(cam_pts[:, 2].max())
    cam_obj.location = cam_obj.matrix_world @ mathutils.Vector((center_x, center_y, max_z + ortho_scale * 2))


def _render_view(cam_obj: Any, output_path: Path, rotation_euler: tuple[float, float, float]) -> None:
    """Rotate the camera and render a single frame."""
    require_blender()
    cam_obj.rotation_euler = rotation_euler
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_stage8(
    template_id: str,
    output_dir: Path,
    resolution: int = 1024,
) -> ThumbnailResult:
    """Render front, side, and 45° thumbnails.

    Camera poses (looking at the model from):
      - Front:  +Z → origin  (rotation (0, 0, 0))
      - Side:   +X → origin  (rotation (0, 0, -90°))
      - 45°:    (+X,+Z) → origin  (rotation (0, 0, -45°))
    """
    require_blender()
    LOG.info("Stage 8 — rendering thumbnails for %s at %d×%d.", template_id, resolution, resolution)

    _configure_render_engine(resolution)
    _clear_lights()
    _add_three_point_lighting()

    cam = _ensure_camera()
    _frame_camera_to_scene(cam)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Front view
    front_path = output_dir / f"{template_id}_front.png"
    _render_view(cam, front_path, (0.0, 0.0, 0.0))

    # Side view (left side)
    side_path = output_dir / f"{template_id}_side.png"
    _render_view(cam, side_path, (0.0, 0.0, math.radians(-90)))

    # 45° view
    angle_path = output_dir / f"{template_id}_angle.png"
    _render_view(cam, angle_path, (0.0, 0.0, math.radians(-45)))

    LOG.info("Stage 8 done — front=%s side=%s angle=%s",
             front_path.name, side_path.name, angle_path.name)

    return ThumbnailResult(
        front=front_path,
        side=side_path,
        angle=angle_path,
        resolution=resolution,
    )
