"""Tests for the quality checker."""

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.deformer.quality_checker import QualityChecker


def test_quality_checker_high_score():
    mesh = trimesh.creation.box(extents=(140.0, 45.0, 30.0))

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="Frame", node_name="Frame")
    scene.add_geometry(
        trimesh.creation.box(extents=(18.0, 8.0, 4.0)),
        geom_name="Bridge", node_name="Bridge",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(50.0, 45.0, 2.0)),
        geom_name="LeftLens", node_name="LeftLens",
        transform=trimesh.transformations.translation_matrix((-30.0, 0.0, 0.0)),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(50.0, 45.0, 2.0)),
        geom_name="RightLens", node_name="RightLens",
        transform=trimesh.transformations.translation_matrix((30.0, 0.0, 0.0)),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(70.0, 4.0, 4.0)),
        geom_name="LeftTemple", node_name="LeftTemple",
        transform=trimesh.transformations.translation_matrix((-70.0, 0.0, -10.0)),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(70.0, 4.0, 4.0)),
        geom_name="RightTemple", node_name="RightTemple",
        transform=trimesh.transformations.translation_matrix((70.0, 0.0, -10.0)),
    )
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


def test_quality_checker_fails_when_required_parts_are_missing():
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(140.0, 45.0, 30.0)),
        geom_name="Frame", node_name="Frame",
    )
    context = DeformationContext(
        template_info=None,
        template_scene=scene,
        descriptor=TemplateDescriptor(
            template_name="incomplete",
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
    report = QualityChecker().evaluate(context)
    assert report.passed is False
    assert any("mesh integrity" in warning.lower() for warning in report.warnings)
