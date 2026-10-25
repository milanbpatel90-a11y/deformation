import json

import numpy as np
import pytest
import trimesh

from backend.exporter.glb_exporter import GLBExporter
from backend.models import FrameMaterial, FrameShape, Measurements


def _measurements():
    return Measurements(
        frame_width=140.0,
        lens_width=53.0,
        lens_height=40.0,
        bridge_width=17.0,
        temple_length=140.0,
        material=FrameMaterial.METAL,
        shape=FrameShape.RECTANGLE,
    )


def _scene():
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(140.0, 42.0, 20.0)), geom_name="Frame", node_name="Frame")
    scene.add_geometry(trimesh.creation.box(extents=(2.0, 8.0, 6.0)), geom_name="Bridge", node_name="Bridge")
    scene.add_geometry(
        trimesh.creation.box(extents=(70.0, 4.0, 4.0)),
        geom_name="LeftTemple",
        node_name="LeftTemple",
        transform=trimesh.transformations.translation_matrix((-45.0, 0.0, -10.0)),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(70.0, 4.0, 4.0)),
        geom_name="RightTemple",
        node_name="RightTemple",
        transform=trimesh.transformations.translation_matrix((45.0, 0.0, -10.0)),
    )
    return scene


def test_export_converts_internal_mm_geometry_to_glb_metres_and_keeps_source_unchanged(tmp_path):
    scene = _scene()
    before = scene.bounds.copy()
    target = tmp_path / "model.glb"

    GLBExporter().export(scene, target, _measurements(), "unit_test")

    assert target.is_file()
    assert np.allclose(scene.bounds, before)
    loaded = trimesh.load(str(target), force="scene")
    assert isinstance(loaded, trimesh.Scene)
    x_width_m = float(loaded.bounds[1, 0] - loaded.bounds[0, 0])
    assert 0.155 <= x_width_m <= 0.165


def test_export_fails_closed_on_non_finite_geometry(tmp_path):
    scene = _scene()
    scene.geometry["Frame"].vertices[0, 0] = np.nan
    target = tmp_path / "must_not_exist.glb"

    with pytest.raises(ValueError, match="NaN or infinite"):
        GLBExporter().export(scene, target, _measurements(), "bad_template")
    assert not target.exists()


def test_sidecar_declares_units_and_writes_anchor_coordinates(tmp_path):
    metadata = GLBExporter()
    anchors = metadata._compute_anchors(_scene(), _measurements(), coordinate_scale=0.001)
    path = tmp_path / "model.metadata.json"
    metadata.export_metadata_json(
        # ExportMetadata values are kept in physical millimetres.
        __import__("backend.models", fromlist=["ExportMetadata"]).ExportMetadata(
            shape="rectangle", material="metal", frame_width=140.0, bridge_width=17.0,
            temple_length=140.0, template_used="unit_test", color="#000000",
        ),
        anchors,
        path,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["units"]["geometry"] == "m"
    assert data["units"]["measurements"] == "mm"
    assert data["units"]["anchors"] == "m"
