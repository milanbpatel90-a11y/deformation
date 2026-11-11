"""Shared runtime container passed through topology-aware deformation stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.models import Measurements, TemplateInfo
from backend.template_matching.feature_extractor import EyewearFeatureSet


@dataclass
class DeformationContext:
    """Mutable runtime state for specialized deformer modules."""

    template_info: TemplateInfo
    template_scene: trimesh.Scene
    descriptor: TemplateDescriptor
    measurements: Measurements
    feature_set: EyewearFeatureSet | None = None
    materials: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.template_scene = deepcopy(self.template_scene)
        self.meshes = self._world_space_geometry(self.template_scene)
        if not self.materials:
            self.materials = {
                name: getattr(mesh.visual, "material", None)
                for name, mesh in self.meshes.items()
            }

    meshes: dict[str, trimesh.Trimesh] = field(init=False, default_factory=dict)

    @staticmethod
    def _world_space_geometry(scene: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
        """{geometry_name: mesh} with each node's transform baked into its
        vertices.

        trimesh.Scene.geometry holds untransformed LOCAL mesh data. Building
        the context straight from it (the previous behaviour) silently
        operates on the wrong local frame for any part whose scene-graph node
        carries a non-identity transform -- confirmed on geometric_metal.glb,
        whose Blender-exported lens node is a ~90 degree rotation that swaps
        the mesh's local Y and Z axes. LensDeformer splits a lens into
        front/back surfaces by Z, so without this bake its "Z" was actually
        the lens's height, not its depth, and every rim-fit/validity check
        failed (see the matching fix and full root-cause note in
        DeformationPipeline._inject_aliases_into_scene, which bakes the same
        way for alias-copied logical names before this method ever sees the
        scene). Keyed by GEOMETRY name (not node name) so existing lookups by
        mesh/part name keep working. A geometry with no scene-graph node of
        its own (an alias-injected logical name such as "LeftLens", added
        directly into scene.geometry) is carried through unchanged here --
        it was already baked at injection time, using its source geometry's
        transform, so re-baking it against a (nonexistent) node of its own
        would be a no-op at best and wrong if it coincidentally matched some
        other node's name.
        """
        baked: dict[str, trimesh.Trimesh] = {}
        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node_name]
            if geom_name in baked:
                continue
            src = scene.geometry.get(geom_name)
            if not isinstance(src, trimesh.Trimesh):
                continue
            mesh = src.copy()
            if not np.allclose(transform, np.eye(4)):
                mesh.apply_transform(transform)
            baked[geom_name] = mesh
        for name, mesh in scene.geometry.items():
            if name not in baked and isinstance(mesh, trimesh.Trimesh):
                baked[name] = mesh
        return baked

    def mesh(self, name: str) -> trimesh.Trimesh:
        mesh = self.meshes.get(name)
        if mesh is None:
            raise KeyError(f"Mesh '{name}' not found in deformation context.")
        return mesh

    def vertex_group_meshes(self, group_name: str) -> list[trimesh.Trimesh]:
        parts = self.descriptor.vertex_groups.get(group_name, [])
        return [self.mesh(part) for part in parts]

    def sync_meshes_to_scene(self) -> trimesh.Scene:
        """Write deformed world-space meshes back to the scene's local geometry.

        Deformation stages operate on independent, world-space copies in
        self.meshes. GLB export reads template_scene and therefore must receive
        those updated meshes, transformed back into each node's local coordinate
        system. Logical aliases are copied back to their original GLB geometry
        entries; alias-only geometry is retained for metadata and anchors.
        """
        aliases = self.descriptor.raw.get("mesh_aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
        aliases = {
            logical: actual
            for logical, actual in aliases.items()
            if isinstance(logical, str) and isinstance(actual, str)
        }
        alias_sources = set(aliases.values())

        node_transforms: dict[str, np.ndarray] = {}
        for node_name in self.template_scene.graph.nodes_geometry:
            transform, geometry_name = self.template_scene.graph[node_name]
            if geometry_name in node_transforms and not np.allclose(
                node_transforms[geometry_name], transform
            ):
                raise ValueError(
                    f"GLB geometry '{geometry_name}' is instanced with different transforms; "
                    "split the instances into separate meshes before deformation."
                )
            node_transforms[geometry_name] = transform

        def to_local(mesh: trimesh.Trimesh, geometry_name: str) -> trimesh.Trimesh:
            result = mesh.copy()
            transform = node_transforms.get(geometry_name)
            if transform is not None and not np.allclose(transform, np.eye(4)):
                result.apply_transform(np.linalg.inv(transform))
            return result

        # Synchronize non-aliased geometry first. Alias sources are updated
        # below from their corresponding logical, independently deformed copy.
        for name, mesh in self.meshes.items():
            if name in aliases or name in alias_sources:
                continue
            if name in self.template_scene.geometry:
                self.template_scene.geometry[name] = to_local(mesh, name)

        for logical_name, actual_name in aliases.items():
            mesh = self.meshes.get(logical_name)
            if mesh is None or actual_name not in self.template_scene.geometry:
                continue
            self.template_scene.geometry[actual_name] = to_local(mesh, actual_name)
            # Keep a world-space logical copy for anchors when it is not itself
            # the source geometry referenced by a scene-graph node.
            if logical_name != actual_name:
                if logical_name in node_transforms:
                    raise ValueError(
                        f"Logical alias '{logical_name}' already has its own GLB node; "
                        "remove the conflicting alias or rename the source mesh."
                    )
                self.template_scene.geometry[logical_name] = mesh.copy()

        return self.template_scene

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
