"""Scene utilities for deformation-safe glTF coordinate handling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh


def bake_scene_world(source: trimesh.Scene) -> trimesh.Scene:
    """Bake a scene's node world transforms into copied geometry vertices."""
    if not isinstance(source, trimesh.Scene):
        source = trimesh.Scene(source)

    result = trimesh.Scene()
    result.metadata = deepcopy(source.metadata)

    instance_counts: dict[str, int] = {}
    for node_name in source.graph.nodes_geometry:
        transform, geometry_name = source.graph.get(node_name)
        geometry = source.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue

        count = instance_counts.get(geometry_name, 0)
        instance_counts[geometry_name] = count + 1
        output_name = geometry_name if count == 0 else f"{geometry_name}__instance_{count}"

        baked = geometry.copy()
        baked.apply_transform(np.asarray(transform, dtype=np.float64))
        result.add_geometry(
            baked,
            geom_name=output_name,
            node_name=output_name,
            transform=np.eye(4),
        )

    if not result.geometry:
        raise ValueError("No mesh geometry found in scene")

    return result


def load_world_baked_scene(path: str | Path) -> trimesh.Scene:
    """Load a GLB and return an identity-transform, world-space meter scene."""
    source = trimesh.load(str(path), force="scene", process=False)
    return bake_scene_world(source)
