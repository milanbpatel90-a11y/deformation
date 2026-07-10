"""Tests for the quality checker."""

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.deformer.quality_checker import QualityChecker


def test_quality_checker_high_score():
    mesh = trimesh.creation.box(extents=(140.0, 45.0, 30.0))

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
        ),
        measurements=None,
    )
    # Give it some fake successful metadata
    context.update_metadata(
        constraint_solver={"failures": [], "corrections": []},
        symmetry_solver={"max_error_mm": 0.1},
    )

    checker = QualityChecker()
    report = checker.evaluate(context)

    assert report.passed is True
    assert report.score >= 90.0
