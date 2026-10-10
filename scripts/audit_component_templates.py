"""Inspect packaged part-deformation GLBs without modifying or splitting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def inspect(path: Path) -> dict:
    scene = trimesh.load(path, force="scene", process=False)
    references: dict[str, list[str]] = {}
    object_names: dict[int, list[str]] = {}
    for geometry_name, mesh in scene.geometry.items():
        object_names.setdefault(id(mesh), []).append(geometry_name)
    for node in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph[node]
        references.setdefault(geometry_name, []).append(node)

    geometries = []
    for geometry_name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        nodes = references.get(geometry_name, [])
        node_records = []
        for node in nodes:
            transform, _ = scene.graph[node]
            world_vertices = trimesh.transform_points(mesh.vertices, transform)
            node_records.append({
                "node": node,
                "transform": transform.tolist(),
                "world_bounds": np.stack((world_vertices.min(axis=0), world_vertices.max(axis=0))).tolist(),
            })
        components = []
        for part in mesh.split(only_watertight=False):
            components.append({
                "vertices": int(len(part.vertices)),
                "faces": int(len(part.faces)),
                "bounds": part.bounds.tolist(),
            })
        visual = mesh.visual
        material = getattr(visual, "material", None)
        uv = getattr(visual, "uv", None)
        normals = getattr(mesh, "vertex_normals", None)
        triangles = mesh.vertices[mesh.faces] if mesh.faces.ndim == 2 and mesh.faces.shape[1:] == (3,) else np.empty((0, 3, 3))
        double_areas = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        ) if len(triangles) else np.empty(0)
        scale = max(float(np.ptp(mesh.vertices, axis=0).max()), np.finfo(float).tiny)
        face_tolerance = np.finfo(float).eps * scale * scale * 32
        faces_valid = bool(
            len(mesh.faces) > 0
            and (not len(mesh.faces) or (mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)))
            and np.isfinite(double_areas).all()
            and np.all(double_areas > face_tolerance)
        )
        geometries.append({
            "geometry": geometry_name,
            "referenced_by_nodes": nodes,
            "shared_geometry_names": object_names[id(mesh)],
            "shared_geometry_reference": len(nodes) > 1 or len(object_names[id(mesh)]) > 1,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "local_bounds": mesh.bounds.tolist(),
            "local_extents": mesh.extents.tolist(),
            "connected_component_count": len(components),
            "components": components,
            "visual_kind": visual.kind,
            "material_name": getattr(material, "name", None),
            "uv_count": None if uv is None else int(len(uv)),
            "normals_count": None if normals is None else int(len(normals)),
            "normals_finite": None if normals is None else bool(np.isfinite(normals).all()),
            "coordinates_finite": bool(np.isfinite(mesh.vertices).all()),
            "faces_valid": faces_valid,
            "nodes": node_records,
        })
    return {
        "asset": str(path),
        "scene_nodes": sorted(scene.graph.nodes_geometry),
        "geometry": geometries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", type=Path, default=[
        Path("templates/geometric_metal.glb"),
        Path("templates/rectangle_plastic.glb"),
    ])
    args = parser.parse_args()
    reports = [inspect(path) for path in args.assets]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
