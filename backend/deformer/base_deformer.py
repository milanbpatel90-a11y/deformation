"""Shared helpers for topology-aware deformer stages."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.deformer.deformation_context import DeformationContext


class BaseDeformer:
    """Small shared base for specialized deformers."""

    stage_name = "deformation"

    @staticmethod
    def collect_vertices(mesh, indices: np.ndarray | None = None) -> np.ndarray:
        vertices = mesh.vertices.copy()
        if indices is None:
            return vertices
        return vertices[np.asarray(indices, dtype=np.int32)].copy()

    @staticmethod
    def apply_falloff(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, 0.0, 1.0)
        return values * values * (3.0 - 2.0 * values)

    @staticmethod
    def preserve_symmetry(vertices: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        updated = vertices.copy()
        target = np.arange(len(updated)) if indices is None else np.asarray(indices, dtype=np.int32)
        if len(target) == 0:
            return updated
        updated[target, 0] -= updated[target, 0].mean()
        return updated

    @staticmethod
    def validate_constraints(result: dict[str, Any]) -> dict[str, Any]:
        result["valid"] = bool(result.get("valid", True))
        return result

    def update_context(self, context: DeformationContext, **values: Any) -> DeformationContext:
        context.update_metadata(**{self.stage_name: values})
        return context
