"""Taubin mesh smoother for targeted junction regions.

Design principles
-----------------
* Taubin smoothing (λ/μ steps) reduces noise without volume-loss shrinkage.
* Smoothing is restricted to explicitly defined junctions (e.g., bridge, rims).
* Non-junction vertices are pinned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import trimesh

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


@dataclass
class SmootherReport:
    """Summary of smoothing operations."""

    passed: bool = True
    smoothed_vertices: int = 0
    iterations: int = 0
    regions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "smoothed_vertices": self.smoothed_vertices,
            "iterations": self.iterations,
            "regions": self.regions,
        }


class MeshSmoother(BaseDeformer):
    """Apply Taubin smoothing only to junction regions."""

    stage_name = "mesh_smoother"

    def __init__(self, lam: float = 0.5, mu: float = -0.53, iterations: int = 10):
        self.lam = lam
        self.mu = mu
        self.iterations = max(0, iterations)

    def apply(self, context: DeformationContext) -> DeformationContext:
        """Run targeted smoothing and return the context with report."""
        report = SmootherReport(iterations=self.iterations)
        
        target_regions = context.descriptor.deformation_regions.get("smooth", [])
        
        if not target_regions:
            # Try to infer some sensible default regions if none specified
            target_regions = ["Bridge", "LeftRim", "RightRim", "LeftTemple", "RightTemple"]
        
        report.regions = list(target_regions)

        for region in target_regions:
            try:
                mesh = context.mesh(region)
                smoothed_count = self._smooth_mesh(mesh)
                report.smoothed_vertices += smoothed_count
            except KeyError:
                pass

        return self.update_context(context, **report.to_dict())

    def _smooth_mesh(self, mesh: trimesh.Trimesh) -> int:
        """Apply Taubin smoothing to boundary vertices — fully vectorised."""
        vertices = mesh.vertices.copy()
        bounds = mesh.bounds
        center = vertices.mean(axis=0)
        extents = bounds[1] - bounds[0]

        # Only smooth outer 20% of bounding box (junction regions)
        dist = np.abs(vertices - center) / (extents / 2.0 + 1e-6)
        mask = np.max(dist, axis=1) > 0.8
        if not np.any(mask):
            return 0

        n_verts = len(vertices)
        if not hasattr(mesh, 'vertex_neighbors') or not mesh.vertex_neighbors:
            return 0

        # Build adjacency indices once. Rebuilding these Python lists for every
        # lambda/mu pass made high-poly templates take minutes to process.
        row_idx, col_idx = self._build_adjacency_indices(mesh.vertex_neighbors)
        neighbor_counts = np.bincount(row_idx, minlength=len(vertices)).astype(np.float64)
        neighbor_counts = np.maximum(neighbor_counts, 1.0)

        mask_idx = np.where(mask)[0]

        for _ in range(self.iterations):
            # Lambda step (shrink)
            lap = self._compute_laplacian_vectorised(vertices, row_idx, col_idx, neighbor_counts)
            vertices[mask_idx] += self.lam * lap[mask_idx]

            # Mu step (inflate)
            lap = self._compute_laplacian_vectorised(vertices, row_idx, col_idx, neighbor_counts)
            vertices[mask_idx] += self.mu * lap[mask_idx]

        mesh.vertices = vertices
        return int(np.sum(mask))

    @staticmethod
    def _compute_laplacian_vectorised(
        vertices: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        neighbor_counts: np.ndarray,
    ) -> np.ndarray:
        """Vectorised umbrella Laplacian using index arrays."""
        laplacian = np.zeros_like(vertices)
        if row_idx.size == 0:
            return laplacian
        np.add.at(laplacian, row_idx, vertices[col_idx])
        laplacian /= neighbor_counts[:, None]
        laplacian -= vertices
        return laplacian

    @staticmethod
    def _build_adjacency_indices(neighbors: list) -> tuple[np.ndarray, np.ndarray]:
        """Flatten trimesh neighbor sets into reusable vectorized indices."""
        edges = [(i, neighbor) for i, values in enumerate(neighbors) for neighbor in values]
        if not edges:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        indices = np.asarray(edges, dtype=np.int32)
        return indices[:, 0], indices[:, 1]
