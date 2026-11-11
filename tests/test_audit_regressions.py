"""Regression tests for the production-safety fixes in the audit branch."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.engine import MeshDeformer
from backend.deformer.quality_checker import QualityReport
from backend.exporter.glb_exporter import GLBExporter
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.api.rim_detection_routes import RimDetectionEngine
from backend.auth import _create_session_token, _read_session_token


def _measurements() -> Measurements:
    return Measurements(
        frame_width=140,
        lens_width=52,
        lens_height=42,
        bridge_width=18,
        temple_length=140,
        rim_thickness=1.2,
        material=FrameMaterial.METAL,
        shape=FrameShape.RECTANGLE,
    )


def _independent_geometry() -> dict[str, trimesh.Trimesh]:
    names = (
        "Frame", "Bridge", "LeftRim", "RightRim",
        "LeftLens", "RightLens", "LeftTemple", "RightTemple",
    )
    return {name: trimesh.creation.box() for name in names}


def _runtime_descriptor_args(geometry):
    groups = {
        "frame": ["Frame"],
        "bridge": ["Bridge"],
        "left_rim": ["LeftRim"],
        "right_rim": ["RightRim"],
        "left_lens": ["LeftLens"],
        "right_lens": ["RightLens"],
        "left_temple": ["LeftTemple"],
        "right_temple": ["RightTemple"],
    }
    return {
        "template_path": Path("fixture.glb"),
        "geometry": geometry,
        "hinges": {"left": object(), "right": object()},
        "rim_loops": {"left_rim": object(), "right_rim": object()},
        "lens_planes": {"left": object(), "right": object()},
        "vertex_groups": groups,
    }


def test_descriptor_rejects_two_logical_parts_on_one_source_mesh():
    geometry = _independent_geometry()
    geometry["Bridge"] = geometry["Frame"]
    loader = DescriptorLoader(Path("templates"))

    with pytest.raises(ValueError, match="same source mesh"):
        loader._validate_runtime_descriptor(**_runtime_descriptor_args(geometry))


def test_descriptor_accepts_independent_required_parts():
    loader = DescriptorLoader(Path("templates"))
    loader._validate_runtime_descriptor(**_runtime_descriptor_args(_independent_geometry()))


def test_auto_inferred_aliases_are_persisted_for_runtime_sync():
    geometry = {
        "large_frame": trimesh.creation.box(extents=[80, 30, 8]),
        "lens_pair": trimesh.creation.box(extents=[50, 25, 4]),
        "left_temple": trimesh.creation.box(extents=[8, 5, 60]),
        "right_temple": trimesh.creation.box(extents=[8, 5, 60]),
    }
    geometry["left_temple"].vertices += [-30, 0, 0]
    geometry["right_temple"].vertices += [30, 0, 0]
    payload = {}

    resolved = DescriptorLoader._resolve_geometry_aliases(geometry, payload)

    assert "Frame" in resolved
    assert "mesh_aliases" in payload
    assert payload["mesh_aliases"]["Frame"] == "large_frame"


def _make_context_with_world_alias():
    source = trimesh.creation.box(extents=[140, 50, 8])
    transform = np.eye(4)
    transform[:3, 3] = [100.0, 0.0, 0.0]
    scene = trimesh.Scene()
    scene.add_geometry(
        source,
        geom_name="source_frame",
        node_name="source_frame_node",
        transform=transform,
    )
    world_alias = source.copy()
    world_alias.apply_transform(transform)
    scene.geometry["Frame"] = world_alias

    descriptor = SimpleNamespace(
        raw={"mesh_aliases": {"Frame": "source_frame"}},
        vertex_groups={},
    )
    context = DeformationContext(
        template_info=SimpleNamespace(),
        template_scene=scene,
        descriptor=descriptor,
        measurements=_measurements(),
    )
    return context


def test_deformed_world_mesh_is_written_back_to_local_scene_geometry():
    context = _make_context_with_world_alias()
    frame = context.mesh("Frame")
    frame.vertices = frame.vertices + np.array([5.0, 0.0, 0.0])

    scene = context.sync_meshes_to_scene()

    # The node remains translated by 100 mm; local vertices carry the 5 mm edit.
    assert np.isclose(scene.geometry["source_frame"].vertices[:, 0].mean(), 5.0)
    # The logical alias remains in world space for anchor calculations.
    assert np.isclose(scene.geometry["Frame"].vertices[:, 0].mean(), 105.0)


def test_glb_export_converts_mm_to_m_and_drops_alias_duplicate(tmp_path):
    context = _make_context_with_world_alias()
    frame = context.mesh("Frame")
    frame.vertices = frame.vertices + np.array([5.0, 0.0, 0.0])
    scene = context.sync_meshes_to_scene()

    output = tmp_path / "deformed.glb"
    GLBExporter().export(scene, output, _measurements(), "fixture")

    loaded = trimesh.load(str(output), force="scene")
    assert len(loaded.geometry) == 1
    assert np.isclose(loaded.bounds[1, 0] - loaded.bounds[0, 0], 0.14, atol=1e-5)
    assert np.isclose(loaded.bounds[:, 0].mean(), 0.105, atol=1e-5)
    # Export must not scale or mutate the source scene in place.
    assert np.isclose(scene.bounds[1, 0] - scene.bounds[0, 0], 140.0, atol=1e-5)


def test_failed_deformation_quality_is_rejected_before_export():
    failed = QualityReport(score=40.0, passed=False, warnings=["missing critical geometry"])
    with pytest.raises(ValueError, match="quality gate failed"):
        DeformationPipeline._require_quality(failed)


def test_multi_view_requires_four_or_five_images():
    pipeline = object.__new__(DeformationPipeline)
    with pytest.raises(ValueError, match="4–5"):
        pipeline.run_from_multiple_images(
            [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
            Path("out.glb"),
        )


def test_rim_polygon_resampling_returns_exact_count_and_closes_perimeter():
    engine = RimDetectionEngine()
    polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    points = engine._resample_polygon(polygon, 8)

    assert points.shape == (8, 2)
    assert len(np.unique(points, axis=0)) >= 4
    assert np.max(points[:, 0]) <= 10
    assert np.max(points[:, 1]) <= 10


def test_measurements_reject_impossible_physical_dimensions():
    with pytest.raises(ValidationError):
        Measurements(
            frame_width=-140,
            lens_width=52,
            lens_height=42,
            bridge_width=18,
            temple_length=140,
        )


def test_session_token_contains_session_version_for_revocation():
    token = _create_session_token("test-user", session_version=9)
    payload = _read_session_token(token)

    assert payload is not None
    assert payload["ver"] == 9
