"""Shared runtime container passed through topology-aware deformation stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

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
        self.meshes = {
            name: mesh
            for name, mesh in self.template_scene.geometry.items()
            if isinstance(mesh, trimesh.Trimesh)
        }
        if not self.materials:
            self.materials = {
                name: getattr(mesh.visual, "material", None)
                for name, mesh in self.meshes.items()
            }

    meshes: dict[str, trimesh.Trimesh] = field(init=False, default_factory=dict)

    def mesh(self, name: str) -> trimesh.Trimesh:
        mesh = self.meshes.get(name)
        if mesh is None:
            raise KeyError(f"Mesh '{name}' not found in deformation context.")
        return mesh

    def distinct_part_problems(self) -> list[str]:
        """Describe any logical parts that are the *same* mesh object (aliasing).

        Part-level deformers assume Frame, Bridge, LeftRim... are separate meshes. When
        several logical names share one object, each stage re-deforms the whole frame.
        """
        names = sorted({part for parts in self.descriptor.vertex_groups.values() for part in parts})
        by_id: dict[int, list[str]] = {}
        for name in names:
            mesh = self.meshes.get(name)
            if mesh is not None:
                by_id.setdefault(id(mesh), []).append(name)
        return [
            "parts share one mesh object: " + ", ".join(group)
            for group in by_id.values()
            if len(group) > 1
        ]

    def assert_distinct_parts(self) -> None:
        problems = self.distinct_part_problems()
        if problems:
            raise ValueError("; ".join(problems))

    def vertex_group_meshes(self, group_name: str) -> list[trimesh.Trimesh]:
        parts = self.descriptor.vertex_groups.get(group_name, [])
        return [self.mesh(part) for part in parts]

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
