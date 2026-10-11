#!/usr/bin/env python3
"""Split the fused rectangle_plastic source GLB into the engine's named parts.

The original asset has misleading node names (e.g. ``Temple_Left`` is really the whole
front frame, ``Temple_Right_Mesh`` holds BOTH temples, ``Frame_Mesh`` is hinge hardware),
works in metres and carries a stray 2x2x2 Blender default cube. So parts are identified
by GEOMETRY, never by name:

    front   : the one component wider than 100 mm
    lenses  : two components, tall (>30 mm) and < 70 mm wide
    temples : two components longer than 100 mm in Z
    hinge   : every remaining small component, attached to the temple on its side

The front is then cut by face-centroid X (Left = -X, front = +Z, Y up, mm, like
geometric_metal.glb):   Bridge |x| < BRIDGE_HALF ; rims BRIDGE_HALF <= |x| < END_X ;
Frame (end pieces) |x| >= END_X.  Parts do not overlap.

Usage: python scripts/split_rectangle_plastic.py [source.glb] [output.glb]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_HALF = 7.0   # mm; keyhole bridge arch spans |x| < 7 on this front
END_X = 62.0        # mm; lens outer edge is ~60.7, end pieces start just outside


def _components(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    comps: list[trimesh.Trimesh] = []
    for node in scene.graph.nodes_geometry:
        matrix, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(matrix)
        if np.allclose(mesh.extents, 2.0, atol=1e-3):    # stray Blender default cube
            continue
        mesh.vertices = mesh.vertices * 1000.0           # metres -> mm
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        comps.extend(mesh.split(only_watertight=False))
    return comps


def _cat(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _sub(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    if not mask.any():
        raise ValueError("empty slice while splitting the front")
    part = mesh.submesh([np.where(mask)[0]], append=True, repair=False)
    part.remove_unreferenced_vertices()
    return part


def build(source: Path) -> trimesh.Scene:
    comps = _components(trimesh.load(source, force="scene"))
    front = [c for c in comps if c.extents[0] > 100]
    temples = [c for c in comps if c.extents[2] > 100]
    lenses = [c for c in comps if c.extents[1] > 30 and c.extents[0] < 70 and c.extents[2] < 20]
    used = {id(c) for c in front + temples + lenses}
    hinge = [c for c in comps if id(c) not in used]
    if len(front) != 1 or len(temples) != 2 or len(lenses) != 2:
        raise ValueError(f"unexpected source layout: front={len(front)} temples={len(temples)} lenses={len(lenses)}")

    side = lambda m: "Left" if m.centroid[0] < 0 else "Right"       # Left = -X
    lens = {side(m): m for m in lenses}
    temple = {side(m): [m] for m in temples}
    for h in hinge:                                                   # hinge hardware rides with its temple
        temple[side(h)].append(h)
    hinge_pts = {s: np.vstack([h.vertices for h in hinge if side(h) == s]) for s in ("Left", "Right")}

    front = front[0]
    cx = front.triangles_center[:, 0]
    bridge = _sub(front, np.abs(cx) < BRIDGE_HALF)
    rim = {"Left": _sub(front, (cx <= -BRIDGE_HALF) & (cx > -END_X)),
           "Right": _sub(front, (cx >= BRIDGE_HALF) & (cx < END_X))}
    frame = _sub(front, np.abs(cx) >= END_X)

    # centre on X=0 (the front is symmetric about the nose)
    dx = -0.5 * (front.bounds[0, 0] + front.bounds[1, 0])
    parts = {"Frame": frame, "Bridge": bridge, "LeftRim": rim["Left"], "RightRim": rim["Right"],
             "LeftLens": lens["Left"], "RightLens": lens["Right"],
             "LeftTemple": _cat(temple["Left"]), "RightTemple": _cat(temple["Right"])}
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        mesh.apply_translation([dx, 0, 0])
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    def empty(name: str, point) -> None:
        m = np.eye(4); m[:3, 3] = point
        scene.graph.update(frame_to=name, frame_from=scene.graph.base_frame, matrix=m)

    for s in ("Left", "Right"):
        empty(f"{s}_Hinge", hinge_pts[s].mean(axis=0) + [dx, 0, 0])
        t = parts[f"{s}Temple"].vertices
        empty(f"{s}_Temple_End", t[np.argmin(t[:, 2])])               # farthest point behind the hinge
    empty("Bridge_Center", parts["Bridge"].bounds.mean(axis=0))
    return scene


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "templates" / "raw" / "rectangle_plastic_original.glb"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "templates" / "rectangle_plastic.glb"
    scene = build(src)
    scene.export(str(out))
    print(f"wrote {out}")
    for n, g in scene.geometry.items():
        print(f"  {n:11s} verts {len(g.vertices):5d} extent(mm) {np.round(g.extents, 1).tolist()} centre {np.round(g.centroid, 1).tolist()}")


if __name__ == "__main__":
    main()
