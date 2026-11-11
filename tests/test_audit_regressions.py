"""Regression tests for the production-safety fixes in the audit branch."""

from pathlib import Path
import json
import struct
import time
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
from backend.template_library.loader import TemplateLibrary
from backend.api.rim_detection_routes import RimDetectionEngine
from backend.auth import (
    SESSION_COOKIE_NAME,
    _create_session_token,
    _db,
    _read_session_token,
    get_current_user_optional,
)
from starlette.requests import Request


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


def test_generic_mesh_names_require_explicit_reviewed_aliases():
    geometry = {
        "large_frame": trimesh.creation.box(extents=[80, 30, 8]),
        "lens_pair": trimesh.creation.box(extents=[50, 25, 4]),
        "left_temple": trimesh.creation.box(extents=[8, 5, 60]),
        "right_temple": trimesh.creation.box(extents=[8, 5, 60]),
    }
    with pytest.raises(ValueError, match="automatic mesh guessing is disabled"):
        DescriptorLoader._resolve_geometry_aliases(geometry, {})


def test_explicit_mesh_aliases_resolve_each_logical_part():
    geometry = {f"source_{name}": mesh for name, mesh in _independent_geometry().items()}
    aliases = {name: f"source_{name}" for name in _independent_geometry()}
    payload = {"mesh_aliases": aliases}

    resolved = DescriptorLoader._resolve_geometry_aliases(geometry, payload)

    assert all(resolved[name] is geometry[f"source_{name}"] for name in aliases)
    assert payload["mesh_aliases"] == aliases


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
    raw = output.read_bytes()
    json_length, _ = struct.unpack("<II", raw[12:20])
    gltf_json = json.loads(raw[20:20 + json_length])
    scene_extras = gltf_json["scenes"][gltf_json.get("scene", 0)]["extras"]["extras"]
    assert scene_extras["units"]["geometry"] == "m"
    assert len(scene_extras["anchors"]["NoseBridge"]) == 3
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


def test_uncalibrated_image_estimates_are_blocked_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ALLOW_UNCALIBRATED_GENERATION", raising=False)
    with pytest.raises(ValueError, match="calibrated physical measurements"):
        DeformationPipeline._require_calibrated_scale(_measurements())


def test_user_supplied_calibrated_measurements_pass_production_gate(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ALLOW_UNCALIBRATED_GENERATION", raising=False)
    measurements = _measurements().model_copy(
        update={"scale_source": "user_supplied", "scale_calibrated": True}
    )
    DeformationPipeline._require_calibrated_scale(measurements)


def test_template_names_cannot_escape_template_directory():
    with pytest.raises(ValueError, match="Invalid template name"):
        TemplateLibrary._validate_template_name("../secrets")


def test_descriptor_loader_rejects_path_traversal_before_reading():
    loader = DescriptorLoader(Path("templates"))
    with pytest.raises(ValueError, match="Invalid template name"):
        loader.load("../secrets")


def test_measurements_reject_impossible_physical_dimensions():
    with pytest.raises(ValidationError):
        Measurements(
            frame_width=-140,
            lens_width=52,
            lens_height=42,
            bridge_width=18,
            temple_length=140,
        )


def test_session_version_revokes_previously_issued_tokens():
    user_id = "audit-session-revocation-test"
    with _db() as connection:
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        connection.execute(
            """
            INSERT INTO users (
                id, username, password_hash, role, is_active, created_at, session_version
            ) VALUES (?, ?, ?, ?, 1, ?, 4)
            """,
            (user_id, user_id, "test-hash", "viewer", int(time.time())),
        )

    def request_with_version(version):
        token = _create_session_token(user_id, session_version=version)
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [
                    (b"cookie", f"{SESSION_COOKIE_NAME}={token}".encode("ascii"))
                ],
            }
        )

    assert get_current_user_optional(request_with_version(4))["id"] == user_id
    assert get_current_user_optional(request_with_version(3)) is None

    with _db() as connection:
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))


def test_session_token_contains_session_version_for_revocation():
    token = _create_session_token("test-user", session_version=9)
    payload = _read_session_token(token)

    assert payload is not None
    assert payload["ver"] == 9
