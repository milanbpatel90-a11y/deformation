"""Integration tests for the Gold Template with the current deformation engine."""

from pathlib import Path

import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.engine import MeshDeformer
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


REQUIRED_MESHES = {
    "Frame",
    "LeftLens",
    "RightLens",
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftTemple",
    "RightTemple",
}


def test_gold_template_loads_and_deforms():
    library = TemplateLibrary()
    assert "gold_template" in library.list_templates()

    info = library.load("gold_template")
    assert Path(info.glb_path).name == "gold_template.glb"
    assert info.shape == FrameShape.GEOMETRIC
    assert info.material == FrameMaterial.METAL

    measurements = Measurements(
        frame_width=145.0,
        lens_width=55.0,
        lens_height=37.0,
        bridge_width=18.0,
        temple_length=155.0,
        rim_thickness=1.2,
        material=FrameMaterial.METAL,
        shape=FrameShape.GEOMETRIC,
        nose_pads=True,
        temple_curve_angle=28.0,
        color="#222222",
    )

    scene = trimesh.load(info.glb_path, force="scene")
    descriptor = DescriptorLoader().load(
        "gold_template",
        measurements=measurements,
        template_info=info,
    )

    context = DeformationContext(
        template_info=info,
        template_scene=scene,
        descriptor=descriptor,
        measurements=measurements,
    )

    missing = REQUIRED_MESHES.difference(context.meshes)
    assert not missing, f"Gold Template is missing required meshes: {sorted(missing)}"

    deformer = MeshDeformer(
        scene,
        info.dimensions,
        rim_pull_strength=library.rim_pull_strength("gold_template"),
    )
    result, quality = deformer.deform(context)

    assert quality.passed, quality.to_dict()
    assert result.meshes
    assert not REQUIRED_MESHES.difference(result.meshes)
