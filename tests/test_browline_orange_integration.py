"""Integration tests for the Browline Orange multi-part template.

Run with: pytest -q tests/test_browline_orange_integration.py
The binary GLB must be present at templates/browline_orange.glb.
"""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from backend.deformer.bridge_deformer import BridgeDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.lens_deformer import LensDeformer
from backend.deformer.rim_deformer import RimDeformer
from backend.deformer.temple_deformer import TempleDeformer
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


def _measurements(**overrides):
    values = dict(
        frame_width=140.36,
        lens_width=50.71,
        lens_height=36.8,
        bridge_width=15.63,
        temple_length=136.94,
        rim_thickness=2.7,
        temple_curve_angle=8.0,
        nose_pad_distance=20.49,
        nose_pad_angle=15.0,
        nose_pad_height=9.61,
        material=FrameMaterial.PLASTIC,
        shape=FrameShape.BROWLINE,
    )
    values.update(overrides)
    return Measurements(**values)


def _context(measurements=None):
    library = TemplateLibrary(TEMPLATES)
    info = library.load("browline_orange")
    descriptor = DescriptorLoader(TEMPLATES).load(
        "browline_orange", measurements=measurements or _measurements(), template_info=info
    )
    scene = trimesh.load(info.glb_path, force="scene")
    return DeformationContext(
        template_info=info,
        template_scene=scene,
        descriptor=descriptor,
        measurements=measurements or _measurements(),
    )


def test_library_and_descriptor_load_without_aggregate_frame():
    ctx = _context()
    assert "Frame" not in ctx.meshes
    assert ctx.descriptor.vertex_groups["left_rim"] == ["LeftRim"]
    assert ctx.descriptor.vertex_groups["right_rim"] == ["RightRim"]
    assert ctx.descriptor.vertex_groups.get("frame", []) == []
    assert np.allclose(ctx.descriptor.hinges["left"].pivot, [-67.978, 43.624, 91.731], atol=0.1)
    assert np.allclose(ctx.descriptor.hinges["right"].pivot, [67.978, 43.624, 91.731], atol=0.1)


def test_native_measurements_do_not_inflate_or_shift_rims():
    ctx = _context()
    before = {name: ctx.mesh(name).bounds.copy() for name in ("LeftRim", "RightRim")}
    RimDeformer(contour_strength=0.2).apply(ctx)
    for name in before:
        after = ctx.mesh(name).bounds
        assert np.isfinite(after).all()
        assert np.max(np.abs(after - before[name])) < 15.0
    assert ctx.metadata["rim_deformation"]["applied"] is True


def test_bridge_deformation_runs_without_aggregate_frame():
    ctx = _context(_measurements(bridge_width=18.0))
    before = ctx.mesh("Bridge").vertices.copy()
    result = BridgeDeformer().apply(ctx)
    after = ctx.mesh("Bridge").vertices
    assert np.isfinite(after).all()
    assert not np.allclose(before, after)
    assert result.metadata["bridge_deformation"]["applied"] is True


def test_lens_positioning_remains_within_rim_envelopes():
    ctx = _context(_measurements(lens_width=52.0, lens_height=38.0))
    LensDeformer().apply(ctx)
    for side in ("left", "right"):
        lens = ctx.mesh("LeftLens" if side == "left" else "RightLens")
        rim = ctx.mesh("LeftRim" if side == "left" else "RightRim")
        assert np.isfinite(lens.vertices).all()
        assert lens.bounds[0, 0] >= rim.bounds[0, 0] - 3.0
        assert lens.bounds[1, 0] <= rim.bounds[1, 0] + 3.0


def test_temple_pivots_are_fixed_and_sides_remain_finite():
    ctx = _context(_measurements(temple_length=145.0))
    for side, part in (("left", "LeftTemple"), ("right", "RightTemple")):
        before = ctx.mesh(part).vertices.copy()
        pivot = ctx.descriptor.hinges[side].pivot.copy()
        TempleDeformer().apply(ctx)
        after = ctx.mesh(part).vertices
        assert np.isfinite(after).all()
        nearest = np.argsort(np.linalg.norm(before - pivot, axis=1))[:8]
        assert np.max(np.linalg.norm(after[nearest] - before[nearest], axis=1)) < 2.0


def test_deformed_scene_exports_and_round_trips_as_glb(tmp_path):
    ctx = _context(_measurements(bridge_width=18.0, lens_width=52.0, temple_length=145.0))
    RimDeformer(contour_strength=0.2).apply(ctx)
    BridgeDeformer().apply(ctx)
    LensDeformer().apply(ctx)
    # Export the actual context meshes as named independent geometry.
    scene = trimesh.Scene()
    for name, mesh in ctx.meshes.items():
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    target = tmp_path / "browline_orange_test.glb"
    scene.export(target, file_type="glb")
    loaded = trimesh.load(target, force="scene")
    assert set(("LeftRim", "RightRim", "Bridge", "LeftLens", "RightLens")).issubset(loaded.geometry)
    bounds = trimesh.util.concatenate(list(loaded.geometry.values())).bounds
    assert np.isfinite(bounds).all()
    assert 0.10 < (bounds[1, 0] - bounds[0, 0]) / 1000.0 < 0.20
