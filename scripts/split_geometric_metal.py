"""Split templates/geometric_metal.glb into the 8 physically separate named parts
the deformation engine expects (see GLB_MESH_RENAMING_GUIDE.md).

Replaces the old `mesh_aliases` workaround, which made Frame/Bridge/LeftRim/RightRim
all the *same* Python object.

What it does
  * bakes the glTF node transforms into the vertices (engine reads raw vertices:
    X = width, Y = height (up), Z = depth) and converts metres -> millimetres
  * Plane_glasses_mat_0 (61,872 verts, 43 loose pieces) is split by face position:
      the two large front shells  -> LeftRim / Bridge / RightRim  (X cuts)
      every other piece           -> Frame (hinges, nose-pad parts, end pieces)
  * Cube.002_glass_mat_0          -> LeftLens / RightLens (by side of X=0)
  * Plane.001 / Plane.002         -> LeftTemple / RightTemple
  "Left" = -X, matching descriptor_loader._infer_hinge_pivot.

Usage:  python scripts/split_geometric_metal.py [--src SRC] [--out OUT]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_HALF_MM = 10.3        # inner lens edge is at |x| ~ 10.2 mm
BIG_COMPONENT_VERTS = 5000   # the two front shells are ~18k verts; everything else < 2.3k


def _baked_world_meshes(scene: trimesh.Scene, scale: float) -> dict[str, trimesh.Trimesh]:
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        out[geom_name] = mesh
    return out


def _vertex_components(mesh: trimesh.Trimesh) -> np.ndarray:
    f, n = mesh.faces, len(mesh.vertices)
    r = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    c = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    _, labels = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    return labels


def _faces(mesh: trimesh.Trimesh, mask: np.ndarray, name: str) -> trimesh.Trimesh:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError(f"part {name} would be empty")
    part = mesh.submesh([idx], append=True)
    part.metadata["name"] = name
    return part


def split(src: Path, out: Path) -> dict[str, trimesh.Trimesh]:
    scene = trimesh.load(src, force="scene")
    scale = 1000.0 if str(scene.units) == "meters" else 1.0
    world = _baked_world_meshes(scene, scale)
    for required in ("Plane_glasses_mat_0", "Cube.002_glass_mat_0", "Plane.001_glasses_mat_0", "Plane.002_glasses_mat_0"):
        if required not in world:
            raise RuntimeError(f"{src} does not look like the original geometric_metal.glb (missing {required})")

    parts: dict[str, trimesh.Trimesh] = {}

    # ---- frame body ------------------------------------------------------
    body = world["Plane_glasses_mat_0"]
    labels = _vertex_components(body)
    sizes = np.bincount(labels[body.faces[:, 0]], minlength=labels.max() + 1)
    vert_sizes = np.bincount(labels, minlength=labels.max() + 1)
    big = vert_sizes >= BIG_COMPONENT_VERTS
    face_big = big[labels[body.faces[:, 0]]]
    cx = body.vertices[body.faces][:, :, 0].mean(axis=1)
    parts["LeftRim"] = _faces(body, face_big & (cx < -BRIDGE_HALF_MM), "LeftRim")
    parts["Bridge"] = _faces(body, face_big & (np.abs(cx) <= BRIDGE_HALF_MM), "Bridge")
    parts["RightRim"] = _faces(body, face_big & (cx > BRIDGE_HALF_MM), "RightRim")
    parts["Frame"] = _faces(body, ~face_big, "Frame")

    # ---- lenses ----------------------------------------------------------
    lens = world["Cube.002_glass_mat_0"]
    lab = _vertex_components(lens)
    lcx = lens.vertices[lens.faces][:, :, 0].mean(axis=1)
    parts["LeftLens"] = _faces(lens, lcx < 0, "LeftLens")
    parts["RightLens"] = _faces(lens, lcx > 0, "RightLens")

    # ---- temples ---------------------------------------------------------
    for geom, name, sign in (("Plane.001_glasses_mat_0", "LeftTemple", -1), ("Plane.002_glasses_mat_0", "RightTemple", 1)):
        m = world[geom]
        if np.sign(m.vertices[:, 0].mean()) != sign:
            raise RuntimeError(f"{geom} is on the wrong side for {name}")
        m.metadata["name"] = name
        parts[name] = m

    out_scene = trimesh.Scene()
    for name in ("Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens", "LeftTemple", "RightTemple"):
        out_scene.add_geometry(parts[name], node_name=name, geom_name=name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_scene.export(out)
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "templates" / "geometric_metal.glb"))
    ap.add_argument("--out", default=str(ROOT / "templates" / "geometric_metal.glb"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    backup = ROOT / "templates" / "raw" / "geometric_metal_original.glb"
    if src == out and not backup.exists():
        shutil.copy2(src, backup)
        src = backup
    elif src == out:
        src = backup
    parts = split(src, out)
    for name, m in parts.items():
        b = m.bounds
        print(f"{name:11s} verts={len(m.vertices):6d} faces={len(m.faces):6d} "
              f"x[{b[0][0]:7.1f},{b[1][0]:7.1f}] y[{b[0][1]:6.1f},{b[1][1]:6.1f}] z[{b[0][2]:6.1f},{b[1][2]:6.1f}] mm")
    print("wrote", out)


if __name__ == "__main__":
    main()
