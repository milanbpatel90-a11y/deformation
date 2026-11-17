"""Tests for the quality checker."""

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.deformer.quality_checker import QualityChecker


def test_quality_checker_high_score():
    box = trimesh.creation.box
    scene = trimesh.Scene()
    for name, extents, x in (
        ("Frame", (140.0, 45.0, 8.0), 0.0),
        ("LeftLens", (50.0, 38.0, 2.0), -35.0),
        ("RightLens", (50.0, 38.0, 2.0), 35.0),
        ("Bridge", (18.0, 6.0, 3.0), 0.0),
        ("LeftTemple", (3.0, 6.0, 140.0), -70.0),
        ("RightTemple", (3.0, 6.0, 140.0), 70.0),
    ):
        part = box(extents=extents)
        part.apply_translation((x, 0.0, 0.0))
        scene.add_geometry(part, node_name=name, geom_name=name)

    context = DeformationContext(
        template_info=None,
        template_scene=scene,
        descriptor=TemplateDescriptor(
            template_name="test",
            template_path=None,
            descriptor_path=None,
            metadata_path=None,
            hinges={},
            rim_loops={},
            bridge_center=np.array([0, 0, 0]),
            empty_anchors={},
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


def test_quality_checker_flags_missing_parts():
    scene = trimesh.Scene({"Frame": trimesh.creation.box(extents=(140.0, 45.0, 8.0))})
    context = DeformationContext(
        template_info=None,
        template_scene=scene,
        descriptor=TemplateDescriptor(
            template_name="test", template_path=None, descriptor_path=None, metadata_path=None,
            hinges={}, rim_loops={}, bridge_center=np.array([0, 0, 0]), empty_anchors={},
            temple_axis={}, lens_planes={}, vertex_groups={}, constraints={},
        ),
        measurements=None,
    )
    report = QualityChecker().evaluate(context)
    assert report.score < 90.0
    assert any("Missing critical parts" in w for w in report.warnings)
