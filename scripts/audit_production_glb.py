"""Audit real production GLB assets without modifying them.

This script is intentionally read-only. It parses GLB JSON/BIN structure and
uses trimesh with process=False to measure scene/world geometry as stored.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def parse_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version}")
    if total_length != len(data):
        raise ValueError(f"GLB length mismatch header={total_length} actual={len(data)}")

    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk.rstrip(b" \t\r\n\x00")
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if json_chunk is None:
        raise ValueError("GLB JSON chunk missing")
    return json.loads(json_chunk.decode("utf-8")), bin_chunk


def accessor_array(doc: dict[str, Any], bin_chunk: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    if "bufferView" not in accessor:
        return np.zeros((accessor["count"], TYPE_COMPONENTS[accessor["type"]]), dtype=np.float64)

    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPE[accessor["componentType"]]).newbyteorder("<")
    comps = TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    component_bytes = dtype.itemsize
    packed_stride = component_bytes * comps
    stride = int(view.get("byteStride", packed_stride))
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))

    if stride == packed_stride:
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * comps, offset=start)
        return arr.reshape(count, comps)

    raw = np.empty((count, comps), dtype=dtype)
    for row in range(count):
        row_start = start + row * stride
        raw[row] = np.frombuffer(bin_chunk, dtype=dtype, count=comps, offset=row_start)
    return raw


def node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # glTF matrices are column-major.
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T

    t = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    s = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    q = np.asarray([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm
    w, x, y, z = q

    r = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    sm = np.diag([s[0], s[1], s[2], 1.0])
    tm = np.eye(4)
    tm[:3, 3] = t
    return tm @ r @ sm


def rotation_angle_deg(matrix: np.ndarray) -> float:
    r = matrix[:3, :3]
    scale = np.linalg.norm(r, axis=0)
    scale[scale == 0] = 1.0
    r = r / scale
    trace = float(np.trace(r))
    value = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def graph_world_matrices(doc: dict[str, Any]) -> dict[int, np.ndarray]:
    nodes = doc.get("nodes", [])
    parents: dict[int, int] = {}
    for parent_idx, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[int(child)] = parent_idx

    cache: dict[int, np.ndarray] = {}

    def world(idx: int) -> np.ndarray:
        if idx in cache:
            return cache[idx]
        local = node_local_matrix(nodes[idx])
        parent = parents.get(idx)
        cache[idx] = world(parent) @ local if parent is not None else local
        return cache[idx]

    for idx in range(len(nodes)):
        world(idx)
    return cache


def mesh_primitive_stats(doc: dict[str, Any], bin_chunk: bytes, world: dict[int, np.ndarray]) -> tuple[list[dict], np.ndarray]:
    mesh_to_nodes: dict[int, list[int]] = {}
    for node_idx, node in enumerate(doc.get("nodes", [])):
        if "mesh" in node:
            mesh_to_nodes.setdefault(int(node["mesh"]), []).append(node_idx)

    all_world_positions: list[np.ndarray] = []
    rows: list[dict] = []

    for mesh_idx, mesh in enumerate(doc.get("meshes", [])):
        for prim_idx, primitive in enumerate(mesh.get("primitives", [])):
            attrs = primitive.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            if pos_idx is None:
                continue
            positions = accessor_array(doc, bin_chunk, pos_idx).astype(np.float64)
            normals = accessor_array(doc, bin_chunk, attrs["NORMAL"]).astype(np.float64) if "NORMAL" in attrs else None
            uvs = accessor_array(doc, bin_chunk, attrs["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in attrs else None

            if "indices" in primitive:
                indices = accessor_array(doc, bin_chunk, primitive["indices"]).reshape(-1).astype(np.int64)
            else:
                indices = np.arange(len(positions), dtype=np.int64)
            tri_count = len(indices) // 3 if primitive.get("mode", 4) == 4 else None

            duplicate_vertices = len(positions) - len(np.unique(np.round(positions, 9), axis=0))
            referenced = np.unique(indices)
            unreferenced = max(0, len(positions) - len(referenced))

            degenerate = 0
            if tri_count is not None and len(indices) >= 3:
                tris = positions[indices[:tri_count * 3].reshape(-1, 3)]
                cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
                area2 = np.linalg.norm(cross, axis=1)
                degenerate = int(np.count_nonzero(area2 <= 1e-12))

            normal_info = None
            if normals is not None:
                lengths = np.linalg.norm(normals, axis=1)
                normal_info = {
                    "count": int(len(normals)),
                    "finite": bool(np.isfinite(normals).all()),
                    "zero_count": int(np.count_nonzero(lengths <= 1e-12)),
                    "min_length": float(lengths.min()) if len(lengths) else None,
                    "max_length": float(lengths.max()) if len(lengths) else None,
                }

            local_min = positions.min(axis=0)
            local_max = positions.max(axis=0)
            node_indices = mesh_to_nodes.get(mesh_idx, [])
            instance_bounds = []
            for node_idx in node_indices or [None]:
                wm = world[node_idx] if node_idx is not None else np.eye(4)
                hom = np.c_[positions, np.ones(len(positions))]
                wp = (wm @ hom.T).T[:, :3]
                all_world_positions.append(wp)
                instance_bounds.append({
                    "node": node_idx,
                    "min": wp.min(axis=0).tolist(),
                    "max": wp.max(axis=0).tolist(),
                    "extents": (wp.max(axis=0) - wp.min(axis=0)).tolist(),
                })

            rows.append({
                "mesh_index": mesh_idx,
                "mesh_name": mesh.get("name"),
                "primitive_index": prim_idx,
                "material_index": primitive.get("material"),
                "mode": primitive.get("mode", 4),
                "vertices": int(len(positions)),
                "triangles": int(tri_count) if tri_count is not None else None,
                "indices": int(len(indices)),
                "duplicate_vertex_positions": int(duplicate_vertices),
                "unreferenced_vertices": int(unreferenced),
                "degenerate_triangles": int(degenerate),
                "local_min": local_min.tolist(),
                "local_max": local_max.tolist(),
                "local_extents": (local_max - local_min).tolist(),
                "normals": normal_info,
                "has_uv0": uvs is not None,
                "uv0_finite": bool(np.isfinite(uvs).all()) if uvs is not None else None,
                "instances": instance_bounds,
            })

    combined = np.vstack(all_world_positions) if all_world_positions else np.empty((0, 3))
    return rows, combined


def trimesh_stats(path: Path) -> dict[str, Any]:
    scene = trimesh.load(path, force="scene", process=False)
    geometries = {}
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        faces = np.asarray(geom.faces)
        verts = np.asarray(geom.vertices)
        degenerate = 0
        if len(faces):
            tris = verts[faces]
            area2 = np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=1)
            degenerate = int(np.count_nonzero(area2 <= 1e-12))
        geometries[name] = {
            "vertices": int(len(verts)),
            "triangles": int(len(faces)),
            "bounds": geom.bounds.tolist() if len(verts) else None,
            "extents": geom.extents.tolist() if len(verts) else None,
            "is_watertight": bool(geom.is_watertight),
            "is_winding_consistent": bool(geom.is_winding_consistent),
            "degenerate_triangles": degenerate,
        }

    bounds = scene.bounds
    return {
        "geometry_count": len(geometries),
        "geometries": geometries,
        "scene_bounds": bounds.tolist() if bounds is not None else None,
        "scene_extents": (bounds[1]-bounds[0]).tolist() if bounds is not None else None,
        "metadata": scene.metadata,
    }


def audit(path: Path) -> dict[str, Any]:
    doc, bin_chunk = parse_glb(path)
    world = graph_world_matrices(doc)
    primitives, combined = mesh_primitive_stats(doc, bin_chunk, world)

    nodes = []
    for idx, node in enumerate(doc.get("nodes", [])):
        local = node_local_matrix(node)
        wm = world[idx]
        nodes.append({
            "index": idx,
            "name": node.get("name"),
            "mesh": node.get("mesh"),
            "children": node.get("children", []),
            "translation": node.get("translation"),
            "rotation": node.get("rotation"),
            "scale": node.get("scale"),
            "matrix": node.get("matrix"),
            "local_rotation_deg": rotation_angle_deg(local),
            "world_translation": wm[:3, 3].tolist(),
            "world_rotation_deg": rotation_angle_deg(wm),
        })

    materials = []
    for idx, mat in enumerate(doc.get("materials", [])):
        pbr = mat.get("pbrMetallicRoughness", {})
        materials.append({
            "index": idx,
            "name": mat.get("name"),
            "baseColorFactor": pbr.get("baseColorFactor"),
            "metallicFactor": pbr.get("metallicFactor"),
            "roughnessFactor": pbr.get("roughnessFactor"),
            "baseColorTexture": pbr.get("baseColorTexture"),
            "normalTexture": mat.get("normalTexture"),
            "alphaMode": mat.get("alphaMode", "OPAQUE"),
            "alphaCutoff": mat.get("alphaCutoff"),
            "doubleSided": mat.get("doubleSided", False),
            "extensions": mat.get("extensions"),
            "extras": mat.get("extras"),
        })

    world_bounds = None
    world_extents = None
    if len(combined):
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        world_bounds = [lo.tolist(), hi.tolist()]
        world_extents = (hi-lo).tolist()

    accessors = doc.get("accessors", [])
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "glb_asset": doc.get("asset", {}),
        "scene_index": doc.get("scene", 0),
        "scene_count": len(doc.get("scenes", [])),
        "node_count": len(doc.get("nodes", [])),
        "mesh_count": len(doc.get("meshes", [])),
        "material_count": len(doc.get("materials", [])),
        "texture_count": len(doc.get("textures", [])),
        "image_count": len(doc.get("images", [])),
        "buffer_count": len(doc.get("buffers", [])),
        "buffer_view_count": len(doc.get("bufferViews", [])),
        "accessor_count": len(accessors),
        "accessor_component_types": sorted({a.get("componentType") for a in accessors if "componentType" in a}),
        "primitive_count": len(primitives),
        "vertex_count_sum": sum(p["vertices"] for p in primitives),
        "triangle_count_sum": sum(p["triangles"] or 0 for p in primitives),
        "world_bounds": world_bounds,
        "world_extents": world_extents,
        "nodes": nodes,
        "primitives": primitives,
        "materials": materials,
        "root_extras": doc.get("extras"),
        "scene_extras": [s.get("extras") for s in doc.get("scenes", [])],
        "trimesh": trimesh_stats(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = [audit(Path(p)) for p in args.paths]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    for report in reports:
        print("=" * 80)
        print(report["path"])
        print("size_bytes", report["file_size_bytes"])
        print("asset", json.dumps(report["glb_asset"], sort_keys=True))
        print("nodes", report["node_count"], "meshes", report["mesh_count"], "primitives", report["primitive_count"])
        print("vertices", report["vertex_count_sum"], "triangles", report["triangle_count_sum"])
        print("world_extents", report["world_extents"])
        print("materials", report["material_count"], "textures", report["texture_count"])
        for p in report["primitives"]:
            print(
                "primitive",
                p["mesh_name"],
                "v", p["vertices"],
                "t", p["triangles"],
                "deg", p["degenerate_triangles"],
                "unref", p["unreferenced_vertices"],
                "dup_pos", p["duplicate_vertex_positions"],
                "uv", p["has_uv0"],
            )
        print("nodes_with_large_rotations", [
            (n["index"], n["name"], round(n["local_rotation_deg"], 3))
            for n in report["nodes"]
            if n["local_rotation_deg"] > 45.0
        ])

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
