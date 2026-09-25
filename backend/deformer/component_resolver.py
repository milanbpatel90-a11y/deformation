"""Resolve source GLB meshes into deformation-safe logical components.

No primitive geometry is synthesized. Logical parts are built only from faces
already present in the production GLB. The original source asset is never
modified.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import trimesh

from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.geometry_units import mm_to_m
from backend.models import TemplateInfo

_REQUIRED = (
    "Frame",
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftLens",
    "RightLens",
    "LeftTemple",
    "RightTemple",
)


def _face_partition(mesh: trimesh.Trimesh, face_mask: np.ndarray, label: str) -> trimesh.Trimesh:
    indices = np.flatnonzero(face_mask)
    if len(indices) == 0:
        raise ValueError(f"Cannot build logical component '{label}': no source faces selected")
    part = mesh.submesh([indices], append=True, repair=False)
    if not isinstance(part, trimesh.Trimesh) or len(part.faces) == 0:
        raise ValueError(f"Cannot build logical component '{label}' from source faces")
    part.metadata = deepcopy(mesh.metadata)
    part.metadata["source_face_partition"] = label
    return part


def _split_left_right(mesh: trimesh.Trimesh, center_x: float) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    centers = mesh.triangles_center[:, 0]
    left = _face_partition(mesh, centers < center_x, "LeftLens")
    right = _face_partition(mesh, centers >= center_x, "RightLens")
    return left, right


def _partition_frame(
    mesh: trimesh.Trimesh,
    *,
    center_x: float,
    bridge_width_mm: float,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    centers = mesh.triangles_center[:, 0]
    # Include the physical bridge plus its attachment shoulders.
    # Every original frame face belongs to exactly one partition.
    bridge_half_m = mm_to_m(max(bridge_width_mm * 0.75, 8.0))
    bridge_mask = np.abs(centers - center_x) <= bridge_half_m
    left_mask = centers < center_x - bridge_half_m
    right_mask = ~(bridge_mask | left_mask)

    bridge = _face_partition(mesh, bridge_mask, "Bridge")
    left = _face_partition(mesh, left_mask, "LeftRim")
    right = _face_partition(mesh, right_mask, "RightRim")

    source_faces = len(mesh.faces)
    if len(bridge.faces) + len(left.faces) + len(right.faces) != source_faces:
        raise ValueError("Frame face partition did not preserve every source face")
    return bridge, left, right


def prepare_deformation_scene(
    scene: trimesh.Scene,
    descriptor: TemplateDescriptor,
    template_info: TemplateInfo,
) -> trimesh.Scene:
    """Return a logical-component scene suitable for deformation.

    If the source already contains independently addressable logical geometry,
    it is preserved. For aliased production assets, actual source faces are
    partitioned into logical components. A full Frame proxy is retained only
    for cross-part deformation/quality calculations and is excluded by the
    exporter.
    """
    source = deepcopy(scene)
    geometries = {
        name: mesh
        for name, mesh in source.geometry.items()
        if isinstance(mesh, trimesh.Trimesh)
    }

    if set(_REQUIRED).issubset(geometries):
        source.metadata = deepcopy(source.metadata)
        source.metadata.setdefault("deformation_runtime", {})["frame_proxy"] = None
        return source

    aliases = descriptor.raw.get("_resolved_mesh_aliases")
    if not isinstance(aliases, dict):
        aliases = descriptor.raw.get("mesh_aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}

    missing_aliases = [name for name in _REQUIRED if name not in aliases and name not in geometries]
    if missing_aliases:
        raise ValueError(
            "Production deformation requires resolvable logical components: "
            + ", ".join(sorted(missing_aliases))
        )

    def actual(logical: str) -> trimesh.Trimesh:
        if logical in geometries:
            return geometries[logical]
        raw_name = aliases.get(logical)
        mesh = geometries.get(raw_name)
        if mesh is None:
            raise ValueError(f"Logical component '{logical}' maps to missing geometry '{raw_name}'")
        return mesh

    frame_source = actual("Frame")
    lens_left_source = actual("LeftLens")
    lens_right_source = actual("RightLens")
    left_temple = actual("LeftTemple").copy()
    right_temple = actual("RightTemple").copy()

    center_x = float((frame_source.bounds[0, 0] + frame_source.bounds[1, 0]) * 0.5)

    frame_aliases_same = (
        actual("Bridge") is frame_source
        and actual("LeftRim") is frame_source
        and actual("RightRim") is frame_source
    )
    if frame_aliases_same:
        bridge, left_rim, right_rim = _partition_frame(
            frame_source,
            center_x=center_x,
            bridge_width_mm=template_info.dimensions.bridge_width,
        )
    else:
        bridge = actual("Bridge").copy()
        left_rim = actual("LeftRim").copy()
        right_rim = actual("RightRim").copy()

    if lens_left_source is lens_right_source:
        left_lens, right_lens = _split_left_right(lens_left_source, center_x)
    else:
        left_lens = lens_left_source.copy()
        right_lens = lens_right_source.copy()

    runtime = trimesh.Scene()
    runtime.metadata = deepcopy(source.metadata)
    runtime.metadata["deformation_runtime"] = {
        "frame_proxy": "Frame",
        "source_frame_geometry": aliases.get("Frame", "Frame"),
        "source_lens_geometry": aliases.get("LeftLens", "LeftLens"),
        "partitioned_frame": bool(frame_aliases_same),
        "partitioned_lenses": bool(lens_left_source is lens_right_source),
        "geometry_units": "meter",
    }

    parts = {
        "Frame": frame_source.copy(),
        "Bridge": bridge,
        "LeftRim": left_rim,
        "RightRim": right_rim,
        "LeftLens": left_lens,
        "RightLens": right_lens,
        "LeftTemple": left_temple,
        "RightTemple": right_temple,
    }

    for name, mesh in parts.items():
        runtime.add_geometry(mesh, geom_name=name, node_name=name, transform=np.eye(4))

    return runtime
