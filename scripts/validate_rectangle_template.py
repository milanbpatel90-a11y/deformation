"""Preflight the one-template deformation milestone without changing the GLB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import trimesh


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
TEMPLATE_NAME = "rectangle_plastic"
REQUIRED_PARTS = {
    "Frame", "Bridge", "LeftTemple", "RightTemple", "LeftLens", "RightLens", "LeftRim", "RightRim"
}


def main() -> int:
    descriptor_path = TEMPLATES / "descriptors" / f"{TEMPLATE_NAME}.json"
    glb_path = TEMPLATES / f"{TEMPLATE_NAME}.glb"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    scene = trimesh.load(glb_path, force="scene")

    geometry = set(scene.geometry)
    nodes = set(scene.graph.nodes)
    expected_empties = set(descriptor["empties"].values())
    missing_parts = sorted(REQUIRED_PARTS - geometry)
    missing_empties = sorted(expected_empties - nodes)

    print(f"{'PASS' if not missing_parts and not missing_empties else 'FAIL'}: {glb_path.name} loaded")
    print(f"Geometry: {', '.join(sorted(geometry))}")
    print(f"Named empties found: {', '.join(sorted(expected_empties & nodes)) or 'none'}")
    if missing_parts:
        print(f"Missing semantic mesh parts: {', '.join(missing_parts)}")
    if missing_empties:
        print(f"Missing semantic empties: {', '.join(missing_empties)}")
    return 0 if not missing_parts and not missing_empties else 1


if __name__ == "__main__":
    raise SystemExit(main())
