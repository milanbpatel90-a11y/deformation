"""Fail loudly when a generated GLB violates the deformation contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

try:
    from scripts.align_glb_contract import REQUIRED_NODES, validate_scene
except ModuleNotFoundError:
    from align_glb_contract import REQUIRED_NODES, validate_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    args = parser.parse_args()
    scene = trimesh.load(args.glb, force="scene")
    names = set(scene.geometry) | set(scene.graph.nodes)
    missing = sorted(REQUIRED_NODES - names)
    if missing:
        raise SystemExit(f"FAIL: missing contract nodes: {', '.join(missing)}")
    try:
        report = validate_scene(scene)
    except ValueError as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(f"PASS: {args.glb}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
