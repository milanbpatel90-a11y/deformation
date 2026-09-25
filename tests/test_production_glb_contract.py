"""Regression tests against committed production GLB assets.

These tests deliberately use real repository GLBs. They do not replace the
production assets with generated primitives.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from backend.deformer.descriptor_loader import DescriptorLoader
from backend.exporter.glb_validator import validate_glb
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


TEMPLATES = Path("templates")
OUTPUT = Path("output")
RECTANGLE_GLB = TEMPLATES / "rectangle_plastic.glb"
HISTORICAL_EXPORT = OUTPUT / "4352529fd07b.glb"


def _rectangle_measurements() -> Measurements:
    return Measurements(
        frame_width=140.0,
        lens_width=53.0,
        lens_height=42.0,
        bridge_width=17.2,
        temple_length=145.0,
        rim_thickness=1.2,
        material=FrameMaterial.PLASTIC,
        shape=FrameShape.RECTANGLE,
        nose_pads=False,
    )


def test_real_rectangle_source_is_gltf2_and_world_scale_is_meter_based():
    assert RECTANGLE_GLB.exists(), "real production rectangle GLB is required"
    scene = trimesh.load(RECTANGLE_GLB, force="scene")

    assert scene.geometry
    assert "Frame_Mesh" in scene.graph.nodes

    transform, geom_name = scene.graph.get("Frame_Mesh")
    frame = scene.geometry[geom_name].copy()
    frame.apply_transform(transform)
    width_m = float(frame.bounds[1, 0] - frame.bounds[0, 0])

    # The real committed asset is already glTF-scale metres in world space.
    assert 0.12 <= width_m <= 0.15

    # A known helper cube is also present and must not be allowed to define
    # eyewear bounds during production deformation/export.
    helper_candidates = []
    for node_name in scene.graph.nodes_geometry:
        node_transform, node_geom = scene.graph.get(node_name)
        mesh = scene.geometry[node_geom].copy()
        mesh.apply_transform(node_transform)
        if np.allclose(mesh.extents, [2.0, 2.0, 2.0], atol=1e-6):
            helper_candidates.append(node_name)
    assert helper_candidates, "expected real source helper cube was not found"


def test_real_rectangle_template_is_rejected_until_parts_are_explicitly_mapped():
    library = TemplateLibrary(TEMPLATES)
    info = library.load("rectangle_plastic")
    loader = DescriptorLoader(TEMPLATES)

    with pytest.raises(ValueError, match="mesh_aliases|independent"):
        loader.load(
            "rectangle_plastic",
            measurements=_rectangle_measurements(),
            template_info=info,
            require_independent_parts=True,
        )


def test_real_historical_export_is_detected_as_non_production_glb():
    assert HISTORICAL_EXPORT.exists(), "committed historical GLB is required"
    measurements = Measurements(
        frame_width=135.0,
        lens_width=50.0,
        lens_height=40.0,
        bridge_width=16.2,
        temple_length=160.0,
        material=FrameMaterial.ACETATE,
        shape=FrameShape.GEOMETRIC,
    )

    report = validate_glb(HISTORICAL_EXPORT, measurements)

    assert not report.passed
    joined = " | ".join(report.errors)
    assert "nested extras" in joined
    assert "units metadata" in joined
    assert "NORMAL" in joined
