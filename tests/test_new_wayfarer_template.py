"""Production contract checks for the New Wayfarer deformation template."""

from pathlib import Path
import json

import numpy as np
import trimesh

from backend.deformer.descriptor_loader import DescriptorLoader
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
NAME = "new_wayfarer_acetate"
REQUIRED_PARTS = {
    "Frame",
    "LeftRim",
    "RightRim",
    "Bridge",
    "LeftLens",
    "RightLens",
    "LeftTemple",
    "RightTemple",
    "LeftHinge",
    "RightHinge",
}


def test_new_wayfarer_runtime_contract():
    glb_path = TEMPLATES / f"{NAME}.glb"
    assert glb_path.exists(), (
        "Production New Wayfarer GLB is missing. Add "
        "templates/new_wayfarer_acetate.glb before merging."
    )

    scene = trimesh.load(glb_path, force="scene")
    assert REQUIRED_PARTS.issubset(scene.geometry.keys())
    assert {"LM_LeftHinge", "LM_RightHinge"}.issubset(set(scene.graph.nodes))

    metadata = json.loads((TEMPLATES / f"{NAME}.json").read_text(encoding="utf-8"))
    assert set(metadata["parts"]) == REQUIRED_PARTS

    # Lens deformation in the live engine operates in X/Y and preserves Z as
    # front/back depth. Catch accidental Z-up authoring exports here.
    for part in ("LeftLens", "RightLens"):
        extents = np.asarray(scene.geometry[part].extents, dtype=float)
        assert extents[0] > extents[2]
        assert extents[1] > extents[2]


def test_new_wayfarer_descriptor_loads_with_hinge_overrides():
    library = TemplateLibrary(TEMPLATES)
    info = library.load(NAME)
    measurements = Measurements(
        frame_width=141.3,
        lens_width=55.0,
        lens_height=39.0,
        bridge_width=14.5,
        temple_length=136.6,
        rim_thickness=1.5,
        material=FrameMaterial.ACETATE,
        shape=FrameShape.GEOMETRIC,
        nose_pads=False,
        temple_curve_angle=14.0,
    )

    descriptor = DescriptorLoader(TEMPLATES).load(
        NAME,
        measurements=measurements,
        template_info=info,
    )

    assert descriptor.hinges["left"].part == "LeftTemple"
    assert descriptor.hinges["right"].part == "RightTemple"
    assert "LM_LeftHinge" in descriptor.empty_anchors
    assert "LM_RightHinge" in descriptor.empty_anchors
    np.testing.assert_allclose(
        descriptor.hinges["left"].pivot,
        descriptor.empty_anchors["LM_LeftHinge"],
    )
    np.testing.assert_allclose(
        descriptor.hinges["right"].pivot,
        descriptor.empty_anchors["LM_RightHinge"],
    )
    assert descriptor.vertex_groups["left_rim"] == ["LeftRim"]
    assert descriptor.vertex_groups["right_rim"] == ["RightRim"]
    assert descriptor.constraints["bridge_width"]["min"] == 12.0
    assert descriptor.constraints["bridge_width"]["max"] == 22.0
