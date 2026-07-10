"""Generate procedural geometric_metal.glb template with named parts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder
from trimesh.visual.material import PBRMaterial


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

# Template dimensions in mm (matches geometric_metal.json)
DIMS = {
    "frame_width": 135.0,
    "lens_width": 50.0,
    "lens_height": 46.0,
    "bridge_width": 16.0,
    "temple_length": 135.0,
    "rim_thickness": 1.0,
}


def _hex_material(name: str, color: str, metalness: float, roughness: float) -> PBRMaterial:
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return PBRMaterial(
        name=name,
        baseColorFactor=[r, g, b, 255],
        metallicFactor=metalness,
        roughnessFactor=roughness,
    )


def _geometric_lens_outline(half_w: float, half_h: float, z: float = 0.0) -> np.ndarray:
    """Octagonal geometric lens outline."""
    points = [
        [-half_w * 0.6, half_h, z],
        [-half_w, half_h * 0.5, z],
        [-half_w, -half_h * 0.5, z],
        [-half_w * 0.6, -half_h, z],
        [half_w * 0.6, -half_h, z],
        [half_w, -half_h * 0.5, z],
        [half_w, half_h * 0.5, z],
        [half_w * 0.6, half_h, z],
    ]
    return np.array(points)


def _extrude_rim(outline: np.ndarray, depth: float = 1.5) -> trimesh.Trimesh:
    """Create thin rim mesh from 2D outline."""
    top = outline.copy()
    bottom = outline.copy()
    bottom[:, 2] -= depth
    vertices = np.vstack([top, bottom])
    n = len(outline)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    # Cap faces (fan triangulation)
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
        faces.append([n, n + i + 1, n + i])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _lens_mesh(cx: float, half_w: float, half_h: float) -> trimesh.Trimesh:
    outline = _geometric_lens_outline(half_w, half_h)
    outline[:, 0] += cx
    # Build a thin lens slab from the outline
    top = outline.copy()
    bottom = outline.copy()
    bottom[:, 2] -= 0.8
    vertices = np.vstack([top, bottom])
    n = len(outline)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _rim_mesh(cx: float, half_w: float, half_h: float, thickness: float) -> trimesh.Trimesh:
    outer = _geometric_lens_outline(half_w, half_h)
    inner = _geometric_lens_outline(half_w - thickness * 2, half_h - thickness * 2)
    outer[:, 0] += cx
    inner[:, 0] += cx
    rim = _extrude_rim(outer, depth=thickness * 1.5)
    return rim


def _temple_mesh(side: str, length: float, hinge_x: float) -> trimesh.Trimesh:
    """Straight temple arm with slight downward curve at tip."""
    sections = 12
    points = []
    sign = -1 if side == "left" else 1
    for i in range(sections + 1):
        t = i / sections
        x = hinge_x + sign * length * t
        y = -2.0 * t * t  # gentle curve down
        z = -3.0 - t * 2.0
        points.append([x, y, z])
    path = np.array(points)
    # Thin box segments along path
    meshes = []
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 0.1:
            continue
        bar = box(extents=[seg_len, 1.2, 1.0])
        midpoint = (path[i] + path[i + 1]) / 2.0
        bar.apply_translation(midpoint)
        meshes.append(bar)
    if not meshes:
        return box(extents=[length, 1.2, 1.0])
    return trimesh.util.concatenate(meshes)


def _nose_pads_mesh(bridge_half: float) -> trimesh.Trimesh:
    pad_l = cylinder(radius=2.5, height=1.0, sections=16)
    pad_l.apply_translation([-bridge_half * 0.4, -8.0, 2.0])
    pad_r = cylinder(radius=2.5, height=1.0, sections=16)
    pad_r.apply_translation([bridge_half * 0.4, -8.0, 2.0])
    return trimesh.util.concatenate([pad_l, pad_r])


def _bridge_mesh(bridge_w: float) -> trimesh.Trimesh:
    return box(extents=[bridge_w, 3.0, 2.0])


def build_geometric_metal_scene() -> trimesh.Scene:
    fw = DIMS["frame_width"]
    lw = DIMS["lens_width"]
    lh = DIMS["lens_height"]
    bw = DIMS["bridge_width"]
    tl = DIMS["temple_length"]
    rt = DIMS["rim_thickness"]

    half_lens_w = lw / 2
    half_lens_h = lh / 2
    bridge_half = bw / 2
    left_cx = -(bridge_half + half_lens_w)
    right_cx = bridge_half + half_lens_w
    hinge_left = left_cx - half_lens_w
    hinge_right = right_cx + half_lens_w

    frame_metal = _hex_material("FrameMetal", "#c0c0c0", 1.0, 0.25)
    lens_mat = _hex_material("Lens", "#ffffff", 0.0, 0.05)
    lens_mat.baseColorFactor = [255, 255, 255, 60]

    scene = trimesh.Scene()

    parts = {
        "LeftLens": _lens_mesh(left_cx, half_lens_w * 0.92, half_lens_h * 0.92),
        "RightLens": _lens_mesh(right_cx, half_lens_w * 0.92, half_lens_h * 0.92),
        "LeftRim": _rim_mesh(left_cx, half_lens_w, half_lens_h, rt),
        "RightRim": _rim_mesh(right_cx, half_lens_w, half_lens_h, rt),
        "Bridge": _bridge_mesh(bw),
        "LeftTemple": _temple_mesh("left", tl, hinge_left),
        "RightTemple": _temple_mesh("right", tl, hinge_right),
        "NosePads": _nose_pads_mesh(bridge_half),
        "TempleTips": box(extents=[8, 4, 6]),
    }

    # Frame = combined bounding representation
    frame_parts = [parts["LeftRim"], parts["RightRim"], parts["Bridge"]]
    parts["Frame"] = trimesh.util.concatenate(frame_parts)

    # Position temple tips at ends
    tips = parts["TempleTips"]
    tips.apply_translation([hinge_left - tl, -4, -5])
    tips_r = box(extents=[8, 4, 6])
    tips_r.apply_translation([hinge_right + tl, -4, -5])
    parts["TempleTips"] = trimesh.util.concatenate([tips, tips_r])

    lens_parts = {"LeftLens", "RightLens"}

    for name, mesh in parts.items():
        mesh.visual.material = lens_mat if name in lens_parts else frame_metal
        scene.add_geometry(mesh, geom_name=name, node_name=name)

    return scene


def main() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    scene = build_geometric_metal_scene()
    glb_path = TEMPLATES_DIR / "geometric_metal.glb"
    scene.export(str(glb_path), file_type="glb")
    print(f"Generated: {glb_path}")

    meta_path = TEMPLATES_DIR / "geometric_metal.json"
    if not meta_path.exists():
        meta = {
            "name": "geometric_metal",
            "shape": "geometric",
            "material": "metal",
            "glb_file": "geometric_metal.glb",
            "dimensions": {k: (v if k != "rim_thickness" else 1.0) for k, v in DIMS.items()},
            "parts": list(scene.geometry.keys()),
        }
        meta["dimensions"].update({
            "temple_curve_angle": 28,
            "nose_pad_distance": 16,
            "nose_pad_angle": 15,
            "nose_pad_height": 3,
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Generated: {meta_path}")


if __name__ == "__main__":
    main()
