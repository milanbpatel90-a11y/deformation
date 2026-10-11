"""Template part separation, aliasing detection, and fail-loudly behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.quality_checker import QualityChecker
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline.deformation_pipeline import DeformationPipeline, TemplateNotPreparedError
from backend.template_library.loader import TemplateLibrary

TEMPLATES = Path("templates")
PARTS = ["Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens", "LeftTemple", "RightTemple"]


def _measurements(**kw) -> Measurements:
    base = dict(
        frame_width=140.0, lens_width=52.0, lens_height=44.0, bridge_width=18.0,
        temple_length=140.0, rim_thickness=1.2, material=FrameMaterial.METAL,
        shape=FrameShape.GEOMETRIC, nose_pads=True,
    )
    base.update(kw)
    return Measurements(**base)


def _context(scene: trimesh.Scene | None = None) -> DeformationContext:
    library = TemplateLibrary(TEMPLATES)
    info = library.load("geometric_metal")
    m = _measurements()
    descriptor = DescriptorLoader(TEMPLATES).load("geometric_metal", measurements=m, template_info=info)
    scene = scene if scene is not None else trimesh.load(info.glb_path, force="scene")
    return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=m)


def test_geometric_metal_has_eight_distinct_named_parts():
    scene = trimesh.load(TEMPLATES / "geometric_metal.glb", force="scene")
    assert sorted(scene.geometry) == sorted(PARTS)
    assert len({id(g) for g in scene.geometry.values()}) == len(PARTS)
    for name, mesh in scene.geometry.items():
        assert len(mesh.faces) > 0, name
        assert np.isfinite(mesh.vertices).all(), name


def test_geometric_metal_is_millimetres_and_left_is_negative_x():
    scene = trimesh.load(TEMPLATES / "geometric_metal.glb", force="scene")
    width = scene.bounds[1][0] - scene.bounds[0][0]
    assert 120.0 < width < 160.0, "engine works in mm"
    for side, sign in (("Left", -1), ("Right", 1)):
        for part in ("Rim", "Lens", "Temple"):
            assert np.sign(scene.geometry[side + part].vertices[:, 0].mean()) == sign, side + part
    bridge = scene.geometry["Bridge"]
    assert bridge.bounds[1][0] - bridge.bounds[0][0] < 30.0


def test_descriptor_no_longer_uses_mesh_aliases():
    data = json.loads((TEMPLATES / "descriptors" / "geometric_metal.json").read_text())
    assert "mesh_aliases" not in data


def test_real_template_context_has_no_aliasing():
    ctx = _context()
    assert ctx.distinct_part_problems() == []
    ctx.assert_distinct_parts()


def test_aliased_parts_are_detected_and_fail_quality():
    scene = trimesh.load(TEMPLATES / "geometric_metal.glb", force="scene")
    scene.geometry["Bridge"] = scene.geometry["Frame"]  # the old alias trick
    scene.geometry["LeftRim"] = scene.geometry["Frame"]
    ctx = _context(scene)
    assert any("Bridge" in p and "Frame" in p for p in ctx.distinct_part_problems())
    with pytest.raises(ValueError):
        ctx.assert_distinct_parts()
    report = QualityChecker().evaluate(ctx)
    assert report.passed is False
    assert report.critical


def test_quality_fails_on_frame_sized_bridge_and_nan():
    ctx = _context()
    # stretch the bridge to frame width to emulate the old aliasing outcome
    v = ctx.mesh("Bridge").vertices.copy()
    v[0, 0] = -70.0
    v[1, 0] = 70.0
    ctx.mesh("Bridge").vertices = v
    assert QualityChecker().evaluate(ctx).passed is False

    ctx = _context()
    v = ctx.mesh("LeftLens").vertices.copy()
    v[0, 0] = np.nan
    ctx.mesh("LeftLens").vertices = v
    report = QualityChecker().evaluate(ctx)
    assert report.passed is False and any("NaN" in c for c in report.critical)


def test_unprepared_template_raises_clear_error_not_attribute_error(monkeypatch):
    pipeline = DeformationPipeline(TEMPLATES)

    def boom(*a, **k):
        raise ValueError("Descriptor references unknown geometry part 'Frame'")

    monkeypatch.setattr(pipeline.descriptor_loader, "load", boom)
    with pytest.raises(TemplateNotPreparedError) as exc:
        pipeline.run_from_measurements(_measurements(), "output/_should_not_exist.glb", template_name="geometric_metal")
    assert "geometric_metal" in str(exc.value)
    assert not Path("output/_should_not_exist.glb").exists()
