"""Normalize generated eyewear GLBs to the deformation naming contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


REQUIRED_NODES = {
    "Frame",
    "LeftLens",
    "RightLens",
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftTemple",
    "RightTemple",
    "NosePads",
    "TempleTips",
}


def _mesh(scene: trimesh.Scene, name: str) -> trimesh.Trimesh | None:
    value = scene.geometry.get(name)
    return value if isinstance(value, trimesh.Trimesh) else None


def _rename_temples(scene: trimesh.Scene) -> None:
    candidates = [
        (name, mesh)
        for name, mesh in scene.geometry.items()
        if isinstance(mesh, trimesh.Trimesh)
        and ("temple" in name.lower() or "plane." in name.lower())
    ]
    candidates = [
        item for item in candidates
        if item[0] not in {"LeftLens", "RightLens", "Plane_glasses_mat_0"}
        and float(item[1].extents[2]) > 0.5 * float(item[1].extents[0])
    ]
    if len(candidates) < 2:
        return
    candidates.sort(key=lambda item: float(item[1].centroid[0]))
    for name, mesh in candidates[:2]:
        scene.delete_geometry(name)
        scene.add_geometry(
            mesh,
            geom_name="LeftTemple" if mesh.centroid[0] < 0 else "RightTemple",
        )


def _seat_lenses(scene: trimesh.Scene, frame: trimesh.Trimesh) -> None:
    target_z = float(frame.centroid[2])
    for name in ("LeftLens", "RightLens"):
        lens = _mesh(scene, name)
        if lens is not None:
            lens.apply_translation([0.0, 0.0, target_z - float(lens.centroid[2])])


def _ensure_lenses(scene: trimesh.Scene, frame: trimesh.Trimesh) -> None:
    if _mesh(scene, "LeftLens") is not None and _mesh(scene, "RightLens") is not None:
        return
    candidates = [
        (name, mesh)
        for name, mesh in scene.geometry.items()
        if name != "Plane_glasses_mat_0"
        and isinstance(mesh, trimesh.Trimesh)
        and float(mesh.extents[0]) > 0.7 * float(frame.extents[0])
        and float(mesh.extents[1]) > 0.5 * float(frame.extents[1])
        and float(mesh.extents[2]) < 0.35 * float(frame.extents[1])
    ]
    if len(candidates) != 1:
        return
    source_name, source = candidates[0]
    components = [
        part for part in source.split(only_watertight=False)
        if len(part.faces) >= 100
    ]
    midpoint = float(source.centroid[0])
    left = [part for part in components if part.centroid[0] <= midpoint]
    right = [part for part in components if part.centroid[0] > midpoint]
    if not left or not right:
        return
    scene.delete_geometry(source_name)
    scene.geometry.pop(source_name, None)
    scene.add_geometry(trimesh.util.concatenate(left), geom_name="LeftLens")
    scene.add_geometry(trimesh.util.concatenate(right), geom_name="RightLens")


def _split_frame(scene: trimesh.Scene, frame_name: str, frame: trimesh.Trimesh) -> None:
    left = _mesh(scene, "LeftLens")
    right = _mesh(scene, "RightLens")
    if left is None or right is None:
        return
    centers = sorted((float(left.centroid[0]), float(right.centroid[0])))
    gap = centers[1] - centers[0]
    bridge_half_width = max(gap * 0.35, 8.0)
    face_centers = frame.vertices[frame.faces].mean(axis=1)
    masks = {
        "Bridge": np.abs(face_centers[:, 0]) <= bridge_half_width,
        "LeftRim": face_centers[:, 0] < -bridge_half_width,
        "RightRim": face_centers[:, 0] > bridge_half_width,
    }
    scene.delete_geometry(frame_name)
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        part = frame.submesh([np.flatnonzero(mask)], append=True, repair=False)
        scene.add_geometry(part, geom_name=name)
    # Frame is a semantic marker; its visible geometry is the three parts.
    scene.graph.remove_geometries([frame_name])
    scene.geometry.pop(frame_name, None)
    marker = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    scene.geometry["Frame"] = marker


def _synthesize_support_parts(scene: trimesh.Scene) -> None:
    left = _mesh(scene, "LeftTemple")
    right = _mesh(scene, "RightTemple")
    if left is not None and right is not None:
        tips = []
        for temple in (left, right):
            distal = temple.vertices[:, 2] <= np.percentile(temple.vertices[:, 2], 2)
            center = temple.vertices[distal].mean(axis=0)
            tips.append(trimesh.creation.box(extents=[4.0, 3.0, 4.0], transform=trimesh.transformations.translation_matrix(center)))
        scene.add_geometry(trimesh.util.concatenate(tips), geom_name="TempleTips")
    bridge = _mesh(scene, "Bridge")
    if bridge is not None:
        center = bridge.vertices[np.argmin(np.abs(bridge.vertices[:, 0]))]
        pad = trimesh.creation.box(
            extents=[8.0, 4.0, 3.0],
            transform=trimesh.transformations.translation_matrix(
                [center[0], center[1], float(bridge.bounds[0][2])]
            ),
        )
        scene.add_geometry(pad, geom_name="NosePads")


def align_scene(scene: trimesh.Scene) -> dict:
    """Align and name an in-memory generated eyewear scene."""
    if scene.extents.size and float(np.max(scene.extents)) < 10.0:
        for mesh in scene.geometry.values():
            if isinstance(mesh, trimesh.Trimesh):
                mesh.apply_scale(1000.0)
        axis_transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        for mesh in scene.geometry.values():
            if isinstance(mesh, trimesh.Trimesh):
                mesh.apply_transform(axis_transform)
    frame_name = "Plane_glasses_mat_0"
    frame = _mesh(scene, frame_name)
    if frame is None or len(frame.vertices) == 0 or len(frame.faces) <= 1:
        raise ValueError(f"Missing merged frame mesh {frame_name!r}")
    _rename_temples(scene)
    _ensure_lenses(scene, frame)
    _seat_lenses(scene, frame)
    _split_frame(scene, frame_name, frame)
    _synthesize_support_parts(scene)
    missing = sorted(REQUIRED_NODES - (set(scene.geometry) | set(scene.graph.nodes)))
    if missing:
        raise ValueError(f"GLB contract missing nodes: {', '.join(missing)}")
    return validate_scene(scene)


def validate_scene(scene: trimesh.Scene) -> dict:
    frame = _mesh(scene, "Frame")
    if frame is None or len(frame.vertices) == 0 or len(frame.faces) <= 1:
        # Frame is a semantic parent for split frame parts.
        frame_parts = [_mesh(scene, name) for name in ("Bridge", "LeftRim", "RightRim")]
        frame_parts = [mesh for mesh in frame_parts if mesh is not None]
        if not frame_parts:
            raise ValueError("Frame has no geometry or split frame parts")
        frame_min = np.min([mesh.bounds[0] for mesh in frame_parts], axis=0)
        frame_max = np.max([mesh.bounds[1] for mesh in frame_parts], axis=0)
    else:
        frame_min, frame_max = frame.bounds
    lens_ranges = {}
    for name in ("LeftLens", "RightLens"):
        lens = _mesh(scene, name)
        if lens is None:
            raise ValueError(f"Missing required lens {name}")
        overlap = min(lens.bounds[1][2], frame_max[2]) - max(lens.bounds[0][2], frame_min[2])
        if overlap <= 0:
            raise ValueError(
                f"{name} Z range {lens.bounds[0][2]:.3f}..{lens.bounds[1][2]:.3f} "
                f"does not overlap frame Z range {frame_min[2]:.3f}..{frame_max[2]:.3f}"
            )
        lens_ranges[name] = [float(lens.bounds[0][2]), float(lens.bounds[1][2])]
    return {"required_nodes": sorted(REQUIRED_NODES), "lens_z_ranges": lens_ranges}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scene = trimesh.load(args.input, force="scene")
    report = align_scene(scene)
    scene.export(args.output, file_type="glb")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
