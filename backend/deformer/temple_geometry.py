"""Robust hinge and longitudinal-axis inference for separated temple meshes."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def infer_temple_hinge(
    temple_mesh: trimesh.Trimesh,
    frame_mesh: trimesh.Trimesh | None = None,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a temple's hinge centre and outward longitudinal axis.

    PCA identifies the long axis regardless of whether the GLB stores length
    on X, Y, or Z. The hinge end is selected by proximity to the frame, rather
    than by assuming that the hinge lies at an X bound. Both returned values
    use the same (world/baked) coordinate space as the supplied meshes.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown temple side: {side!r}")
    vertices = np.asarray(temple_mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 3:
        raise ValueError("Temple mesh must contain at least three 3D vertices")

    centre = vertices.mean(axis=0)
    centred = vertices - centre
    covariance = centred.T @ centred / max(len(vertices) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    longitudinal = eigenvectors[:, int(np.argmax(eigenvalues))]
    longitudinal_norm = float(np.linalg.norm(longitudinal))
    if longitudinal_norm < 1e-9:
        raise ValueError("Cannot infer a longitudinal axis from degenerate temple geometry")
    longitudinal /= longitudinal_norm

    projection = centred @ longitudinal
    lower = float(projection.min())
    upper = float(projection.max())
    span = upper - lower
    if span < 1e-6:
        raise ValueError("Temple mesh has no measurable longitudinal extent")

    # The root of this GLB is a short multi-shell hinge connector, not a single
    # flat end cap. Use a wider root band so the pivot lands near the connector's
    # centre/rotation point instead of on its rear-most surface. This also keeps
    # the requested temple length measured from the hinge assembly, not from an
    # arbitrary extreme vertex.
    band = max(span * 0.06, 1e-5)
    low_vertices = vertices[projection <= lower + band]
    high_vertices = vertices[projection >= upper - band]
    if len(low_vertices) == 0 or len(high_vertices) == 0:
        raise ValueError("Could not identify both ends of the temple mesh")

    frame_points = None
    if (
        frame_mesh is not None
        and frame_mesh is not temple_mesh
        and isinstance(frame_mesh, trimesh.Trimesh)
        and len(frame_mesh.vertices) > 0
    ):
        frame_points = np.asarray(frame_mesh.vertices, dtype=np.float64)

    if frame_points is not None:
        tree = cKDTree(frame_points)
        low_distance = float(np.median(tree.query(low_vertices, k=1)[0]))
        high_distance = float(np.median(tree.query(high_vertices, k=1)[0]))
        root_is_low = low_distance <= high_distance
    else:
        # Deterministic fallback when a separate frame mesh is unavailable.
        dominant_component = int(np.argmax(np.abs(longitudinal)))
        root_is_low = longitudinal[dominant_component] >= 0

    root_band = low_vertices if root_is_low else high_vertices
    far_band = high_vertices if root_is_low else low_vertices
    pivot = np.median(root_band, axis=0)
    direction = np.mean(far_band, axis=0) - pivot
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-8:
        direction = longitudinal if root_is_low else -longitudinal
        direction_norm = float(np.linalg.norm(direction))
    axis = direction / max(direction_norm, 1e-12)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return pivot.astype(np.float64), axis.astype(np.float64)
