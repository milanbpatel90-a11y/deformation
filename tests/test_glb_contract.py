from pathlib import Path

import trimesh

from scripts.align_glb_contract import REQUIRED_NODES, align_scene


def test_generated_glb_can_be_normalized_to_contract():
    root = Path(__file__).resolve().parents[1]
    scene = trimesh.load(root / "templates" / "geometric_metal.glb", force="scene")

    report = align_scene(scene)

    assert set(scene.geometry) >= REQUIRED_NODES
    assert report["lens_z_ranges"]["LeftLens"][1] >= report["lens_z_ranges"]["LeftLens"][0]
    assert report["lens_z_ranges"]["RightLens"][1] >= report["lens_z_ranges"]["RightLens"][0]
