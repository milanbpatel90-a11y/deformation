"""Tests for the Taubin mesh smoother using the production geometric GLB."""

from pathlib import Path

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.smoother import MeshSmoother
from backend.models import Measurements
from backend.scene_utils import load_world_baked_scene
from backend.template_library.loader import TemplateLibrary


def test_smoother_preserves_vertex_count():
    library = TemplateLibrary(Path("templates"))
    info = library.load("geometric_metal")
    measurements = Measurements(
        frame_width=135.0,
        lens_width=50.0,
        lens_height=46.0,
        bridge_width=16.0,
        temple_length=135.0,
    )
    descriptor = DescriptorLoader(Path("templates")).load(
        "geometric_metal",
        measurements=measurements,
        template_info=info,
    )
    context = DeformationContext(
        template_info=info,
        template_scene=load_world_baked_scene(info.glb_path),
        descriptor=descriptor,
        measurements=measurements,
    )
    counts_before = {
        name: len(mesh.vertices)
        for name, mesh in context.meshes.items()
    }

    context = MeshSmoother(iterations=5).apply(context)

    counts_after = {
        name: len(mesh.vertices)
        for name, mesh in context.meshes.items()
    }
    assert counts_after == counts_before
