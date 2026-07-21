"""Quality gates for generated geometry and image alignment."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_model(parts: dict[str, Any], source_mask: np.ndarray, texture_path: str, silhouette_iou: float = 1.0, max_triangles: int = 50000) -> dict[str, Any]:
    meshes = list(parts.values())
    triangles = sum(len(mesh.faces) for mesh in meshes)
    watertight = all(mesh.is_watertight for mesh in meshes if len(mesh.faces))
    degenerate = any(np.any(mesh.area_faces <= 1e-12) for mesh in meshes if len(mesh.faces))
    left = parts.get("lens_L")
    right = parts.get("lens_R")
    if left is not None and right is not None:
        symmetry = max(abs(left.extents[i] - right.extents[i]) for i in range(3)) / max(float(max(left.extents.max(), right.extents.max())), 1e-9)
    else:
        symmetry = 0.0
    from PIL import Image
    texture_size = min(Image.open(texture_path).size)
    checks = {"silhouette_iou": bool(silhouette_iou >= 0.85), "watertight": bool(watertight), "no_degenerate_faces": bool(not degenerate), "symmetry_under_3_percent": bool(symmetry < 0.03), "triangle_count_under_50000": bool(triangles <= max_triangles), "texture_resolution_at_least_512": bool(texture_size >= 512)}
    return {"passed": all(checks.values()), "checks": checks, "silhouette_iou": silhouette_iou, "triangle_count": triangles, "symmetry_deviation": symmetry, "texture_resolution": texture_size, "source_mask_shape": list(source_mask.shape)}
