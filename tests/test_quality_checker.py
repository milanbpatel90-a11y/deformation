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

def test_quality_checker_surfaces_lens_aperture_size_mismatch():
    from backend.models import FrameMaterial, FrameShape, Measurements

    def box(extents, center):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return mesh

    parts = {
        "Frame": box((140.0, 24.0, 8.0), (0.0, 29.0, 61.0)),
        "Bridge": box((18.0, 8.0, 3.0), (0.0, 25.0, 63.0)),
        "LeftRim": box((58.0, 42.0, 8.0), (-39.0, 23.0, 61.0)),
        "RightRim": box((58.0, 42.0, 8.0), (39.0, 23.0, 61.0)),
        "LeftLens": box((48.0, 38.0, 2.0), (-36.0, 23.0, 64.0)),
        "RightLens": box((48.0, 38.0, 2.0), (36.0, 23.0, 64.0)),
        "LeftTemple": box((8.0, 10.0, 130.0), (-62.0, 35.0, -5.0)),
        "RightTemple": box((8.0, 10.0, 130.0), (62.0, 35.0, -5.0)),
    }
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    descriptor = TemplateDescriptor(
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
        vertex_groups={
            "frame": ["Frame"], "bridge": ["Bridge"],
            "left_rim": ["LeftRim"], "right_rim": ["RightRim"],
            "left_lens": ["LeftLens"], "right_lens": ["RightLens"],
            "left_temple": ["LeftTemple"], "right_temple": ["RightTemple"],
        },
        constraints={},
    )
    measurements = Measurements(
        frame_width=140.0, lens_width=58.0, lens_height=40.0, bridge_width=18.0,
        temple_length=140.0, rim_thickness=2.0, material=FrameMaterial.METAL,
        shape=FrameShape.RECTANGLE, nose_pads=False,
    )
    context = DeformationContext(None, scene, descriptor, measurements)
    context.update_metadata(
        constraint_solver={"failures": [], "corrections": []},
        symmetry_solver={"max_error_mm": 0.1},
        lens_deformation={"sides": [
            {"side": "left", "dimension_warning": "LeftLens is fitted but too small for the requested measurement."},
        ]},
    )

    report = QualityChecker().evaluate(context)
    assert any("too small for the requested measurement" in warning for warning in report.warnings)
    assert report.passed is False



def test_quality_checker_surfaces_rim_clamp_warning_without_failing_the_gate():
    """A clamped rim opening must be visible in the report; it is a warning, not a failure."""
    from backend.models import FrameMaterial, FrameShape, Measurements

    def box(extents, center):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return mesh

    parts = {
        "Frame": box((140.0, 24.0, 8.0), (0.0, 29.0, 61.0)),
        "Bridge": box((18.0, 8.0, 3.0), (0.0, 25.0, 63.0)),
        "LeftRim": box((58.0, 42.0, 8.0), (-39.0, 23.0, 61.0)),
        "RightRim": box((58.0, 42.0, 8.0), (39.0, 23.0, 61.0)),
        "LeftLens": box((50.0, 38.0, 2.0), (-36.0, 23.0, 64.0)),
        "RightLens": box((50.0, 38.0, 2.0), (36.0, 23.0, 64.0)),
        "LeftTemple": box((8.0, 10.0, 130.0), (-62.0, 35.0, -5.0)),
        "RightTemple": box((8.0, 10.0, 130.0), (62.0, 35.0, -5.0)),
    }
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    descriptor = TemplateDescriptor(
        template_name="test", template_path=None, descriptor_path=None, metadata_path=None,
        hinges={}, rim_loops={}, bridge_center=np.array([0, 0, 0]), empty_anchors={}, temple_axis={},
        lens_planes={},
        vertex_groups={
            "frame": ["Frame"], "bridge": ["Bridge"],
            "left_rim": ["LeftRim"], "right_rim": ["RightRim"],
            "left_lens": ["LeftLens"], "right_lens": ["RightLens"],
            "left_temple": ["LeftTemple"], "right_temple": ["RightTemple"],
        },
        constraints={},
    )
    measurements = Measurements(
        frame_width=140.0, lens_width=52.0, lens_height=40.0, bridge_width=18.0,
        temple_length=140.0, rim_thickness=2.0, material=FrameMaterial.METAL,
        shape=FrameShape.RECTANGLE, nose_pads=False,
    )
    context = DeformationContext(None, scene, descriptor, measurements)
    context.update_metadata(
        constraint_solver={"failures": [], "corrections": []},
        symmetry_solver={"max_error_mm": 0.1},
        rim_deformation={"sides": [
            {"side": "left", "warning": "LeftRim opening limited to 51.8x35.6 mm (wanted 58.7x40.7 mm)."},
            {"side": "right", "warning": None},
        ]},
    )

    report = QualityChecker().evaluate(context)
    assert any("LeftRim opening limited" in w for w in report.warnings)
    assert not any("RightRim" in w for w in report.warnings)
    assert report.passed is True, report.to_dict()
