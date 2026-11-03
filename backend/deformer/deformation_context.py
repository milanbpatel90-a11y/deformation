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

    def sync_scene_from_meshes(self) -> trimesh.Scene:
        """Rebuild the export scene from the current deformed logical part meshes.

        Deformers mutate independent world-space meshes in this context. The
        source GLB scene graph still contains pre-deformation geometry, so returning
        it directly can silently export the original template. Rebuild a clean
        scene with logical part names and identity transforms so mesh edits,
        world-space positions, and materials are serialized.
        """
        logical_parts: list[str] = []
        seen: set[str] = set()
        for parts in self.descriptor.vertex_groups.values():
            for name in parts:
                if name not in seen:
                    logical_parts.append(name)
                    seen.add(name)

        raw_aliases = self.descriptor.raw.get("mesh_aliases", {})
        aliased_source_meshes = set(raw_aliases.values()) if isinstance(raw_aliases, dict) else set()
        # Retain accessories not covered by the deformation descriptors (hinge
        # pins, screws, decorative pieces), but do not export source meshes again
        # when they have already been copied under logical part aliases.
        export_names = list(logical_parts)
        export_names.extend(
            name for name in self.meshes
            if name not in seen and name not in aliased_source_meshes
        )

        export_scene = trimesh.Scene()
        for name in export_names:
            mesh = self.meshes.get(name)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 3 or len(mesh.faces) < 1:
                continue
            export_scene.add_geometry(mesh.copy(), geom_name=name, node_name=name)

        if not export_scene.geometry:
            raise ValueError(
                f"Template '{self.descriptor.template_name}' has no exportable logical part meshes."
            )
        export_scene.metadata = deepcopy(self.template_scene.metadata or {})
        self.template_scene = export_scene
        return self.template_scene

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
