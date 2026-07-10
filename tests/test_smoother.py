"""Tests for the Taubin mesh smoother."""

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.deformer.smoother import MeshSmoother


def test_smoother_preserves_vertex_count():
    mesh = trimesh.creation.box()
    initial_count = len(mesh.vertices)

    context = DeformationContext(
        template_info=None,
        template_scene=trimesh.Scene({"Frame": mesh}),
        descriptor=TemplateDescriptor(
            template_name="test",
            template_path=None,
            descriptor_path=None,
            metadata_path=None,
            hinges={},
            rim_loops={},
            bridge_center=np.array([0, 0, 0]),
            temple_axis={},
            lens_planes={},
            vertex_groups={},
            constraints={},
            deformation_regions={"smooth": ["Frame"]},
        ),
        measurements=None,
    )

    smoother = MeshSmoother(iterations=5)
    context = smoother.apply(context)

    assert len(context.mesh("Frame").vertices) == initial_count
