"""Tests for the quality checker."""

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor
from backend.deformer.quality_checker import QualityChecker


def _context_with_meshes(meshes):
    return DeformationContext(
        template_info=None,
        template_scene=trimesh.Scene(meshes),
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
            constraints={
                "frame_width": {"min": 100.0, "max": 180.0},
                "bridge_width": {"min": 14.0, "max": 24.0},
            },
        ),
        measurements=None,
    )


def _parts(frame=None, bridge=None):
    left_rim = trimesh.creation.box(extents=(50.0, 45.0, 3.0))
    right_rim = left_rim.copy()
    left_rim.apply_translation([-30.0, 0.0, 0.0])
    right_rim.apply_translation([30.0, 0.0, 0.0])
    left_lens = trimesh.creation.box(extents=(45.0, 40.0, 2.0))
    right_lens = left_lens.copy()
    left_lens.apply_translation([-30.0, 0.0, 0.0])
    right_lens.apply_translation([30.0, 0.0, 0.0])
    left_temple = trimesh.creation.box(extents=(5.0, 130.0, 4.0))
    right_temple = left_temple.copy()
    left_temple.apply_translation([-70.0, 0.0, 0.0])
    right_temple.apply_translation([70.0, 0.0, 0.0])
    return {
        "Frame": frame if frame is not None else trimesh.creation.box(extents=(140.0, 45.0, 30.0)),
        "Bridge": bridge if bridge is not None else trimesh.creation.box(extents=(16.0, 3.0, 2.0)),
        "LeftRim": left_rim,
        "RightRim": right_rim,
        "LeftLens": left_lens,
        "RightLens": right_lens,
        "LeftTemple": left_temple,
        "RightTemple": right_temple,
    }


def test_quality_checker_high_score():
    context = _context_with_meshes(_parts())

    context.update_metadata(
        constraint_solver={"failures": [], "corrections": []},
        symmetry_solver={"max_error_mm": 0.1},
    )

    report = QualityChecker().evaluate(context)

    assert report.passed is True
    assert report.score >= 90.0


def test_quality_checker_rejects_missing_logical_parts():
    meshes = _parts()
    del meshes["RightLens"]
    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("RightLens" in warning for warning in report.warnings)


def test_quality_checker_rejects_shared_mesh_object():
    shared = trimesh.creation.box(extents=(140.0, 45.0, 30.0))
    meshes = _parts()
    meshes["Frame"] = shared
    meshes["Bridge"] = shared

    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("identical geometry" in warning for warning in report.warnings)


def test_quality_checker_rejects_duplicate_geometry_copy():
    shared_content = trimesh.creation.box(extents=(10.0, 8.0, 2.0))
    meshes = _parts()
    meshes["LeftRim"] = shared_content
    meshes["RightRim"] = shared_content.copy()

    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("identical geometry" in warning for warning in report.warnings)


def test_quality_checker_rejects_implausible_bridge_by_descriptor_limits():
    meshes = _parts(bridge=trimesh.creation.box(extents=(41.0, 3.0, 2.0)))
    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("Bridge X extent" in warning for warning in report.warnings)


def test_quality_checker_uses_declared_mm_dimensions_for_metre_glb_geometry():
    meshes = _parts(
        frame=trimesh.creation.box(extents=(0.140, 0.045, 0.030)),
        bridge=trimesh.creation.box(extents=(0.016, 0.003, 0.002)),
    )
    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is True


def test_quality_checker_rejects_nonfinite_and_degenerate_geometry():
    meshes = _parts()
    broken = trimesh.creation.box()
    broken.vertices[0, 0] = np.nan
    meshes["LeftRim"] = broken

    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("coordinates contain NaN or infinity" in warning for warning in report.warnings)


def test_quality_checker_rejects_degenerate_faces():
    meshes = _parts()
    broken = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=[[0, 1, 2]], process=False)
    meshes["LeftRim"] = broken

    report = QualityChecker().evaluate(_context_with_meshes(meshes))

    assert report.passed is False
    assert any("degenerate triangles" in warning for warning in report.warnings)
