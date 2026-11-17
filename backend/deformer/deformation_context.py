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
    # Per-side in-plane warps (callables: (N,3) vertices -> (N,3)) published by RimDeformer so
    # later stages (lenses) move with the rim instead of being re-fitted to it.
    rim_warps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.template_scene = deepcopy(self.template_scene)
        self.meshes = self._bake_node_transforms_in_place(self.template_scene)
        # Pristine (baked) template vertices, so stages can tell what has actually moved.
        self.original_vertices = {name: mesh.vertices.copy() for name, mesh in self.meshes.items()}
        if not self.materials:
            self.materials = {
                name: getattr(mesh.visual, "material", None)
                for name, mesh in self.meshes.items()
            }

    meshes: dict[str, trimesh.Trimesh] = field(init=False, default_factory=dict)
    original_vertices: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    @staticmethod
    def _bake_node_transforms_in_place(scene: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
        """Bake each node's world transform into its geometry, IN PLACE, and
        return {geometry_name: mesh} for the scene's own Trimesh objects.

        trimesh.Scene.geometry holds untransformed LOCAL mesh data, so reading
        it directly operates in the wrong frame for any part whose node has a
        non-identity transform (confirmed on geometric_metal.glb: the lens
        node is a ~90 degree rotation swapping local Y/Z, and LensDeformer
        splits front/back by Z -- see DeformationPipeline._inject_aliases_
        into_scene for the full note).

        This MUST mutate the scene's own geometry rather than return copies:
        deformers edit ``context.meshes[...]`` and the pipeline exports
        ``context.template_scene``. If the two are different objects the
        exported GLB is silently the *undeformed* template (the previous
        version returned ``.copy()`` and did exactly that: every directly
        named part moved 0.000 mm). After baking, the node's transform is
        reset to identity so export does not apply it a second time.

        A geometry instanced by more than one node cannot be baked into a
        single shared mesh without changing the others, so it is left as-is.
        Alias-injected logical names (e.g. "LeftLens") have no node of their
        own and were already baked at injection time.
        """
        refs: dict[str, list[tuple[str, np.ndarray]]] = {}
        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node_name]
            refs.setdefault(geom_name, []).append((node_name, transform))
        for geom_name, nodes in refs.items():
            if len(nodes) != 1:
                continue
            node_name, transform = nodes[0]
            mesh = scene.geometry.get(geom_name)
            if not isinstance(mesh, trimesh.Trimesh) or np.allclose(transform, np.eye(4)):
                continue
            mesh.apply_transform(transform)
            scene.graph.update(frame_to=node_name, frame_from=scene.graph.base_frame, matrix=np.eye(4))
        return {n: m for n, m in scene.geometry.items() if isinstance(m, trimesh.Trimesh)}

    def mesh(self, name: str) -> trimesh.Trimesh:
        mesh = self.meshes.get(name)
        if mesh is None:
            raise KeyError(f"Mesh '{name}' not found in deformation context.")
        return mesh

    def vertex_group_meshes(self, group_name: str) -> list[trimesh.Trimesh]:
        parts = self.descriptor.vertex_groups.get(group_name, [])
        return [self.mesh(part) for part in parts]

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
