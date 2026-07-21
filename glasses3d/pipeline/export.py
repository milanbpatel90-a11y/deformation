"""Export named parts as a GLB with a try-on-friendly scene hierarchy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh


def export_glb(model: dict[str, Any], output_path: str | Path, materials: dict[str, Any] | None = None) -> Path:
    del materials
    scene = trimesh.Scene()
    root = scene.graph.base_frame
    for name, mesh in model["parts"].items():
        node = name
        transform = trimesh.transformations.translation_matrix(model.get("pivots", {}).get("hinge_L" if name == "temple_L" else "hinge_R" if name == "temple_R" else "bridge", [0, 0, 0])) if name.startswith("temple_") else None
        scene.add_geometry(mesh, node_name=node, geom_name=name, transform=transform)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output, file_type="glb")
    return output
