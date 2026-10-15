"""RimDeformer must carry the requested lens size into the rim opening.

Regression: the old implementation selected "front"/"back" rim vertices as those at exactly the
global max/min Z. On the real split templates that is ONE vertex (of 15,554 on geometric_metal),
so lens_width / lens_height never reached the rims. These tests measure the opening from the
resulting mesh, independently of the deformer's own metadata.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.lens_deformer import LensDeformer
from backend.deformer.rim_deformer import RimDeformer
from backend.models import FrameMaterial, FrameShape, LensContour, Measurements
from backend.template_library.loader import TemplateLibrary

TEMPLATES = Path("templates")
CLEARANCE = 0.35  # RimDeformer.lens_clearance_mm == LensDeformer.rim_inset
CONTOUR = LensContour(
    left=[[0.04, 0.18], [0.10, 0.06], [0.38, 0.04], [0.46, 0.20], [0.46, 0.76], [0.36, 0.92], [0.10, 0.90], [0.03, 0.70]],
    right=[[0.54, 0.20], [0.62, 0.04], [0.90, 0.06], [0.96, 0.18], [0.97, 0.70], [0.90, 0.90], [0.64, 0.92], [0.54, 0.76]],
)


def _context(template: str, lens_w: float, lens_h: float) -> DeformationContext:
    info = TemplateLibrary(TEMPLATES).load(template)
    m = Measurements(
        frame_width=148.0, lens_width=lens_w, lens_height=lens_h, bridge_width=18.0, temple_length=140.0,
        rim_thickness=2.2, material=FrameMaterial.METAL, shape=FrameShape.RECTANGLE, nose_pads=True,
    )
    descriptor = DescriptorLoader(TEMPLATES).load(template, measurements=m, template_info=info)
    scene = trimesh.load(info.glb_path, force="scene")
    return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=m)


def _opening(ctx: DeformationContext, side: str) -> Polygon:
    helper = LensDeformer()
    selection = helper._collect_lenses(ctx, side)
    return Polygon(helper._extract_rim_boundary(ctx, selection))


def _size(poly: Polygon) -> tuple[float, float]:
    x0, y0, x1, y1 = poly.bounds
    return x1 - x0, y1 - y0


@lru_cache(maxsize=None)
def _run(template: str, lens_w: float, lens_h: float, with_contour: bool):
    """Deform once and keep everything the tests need (the geometric_metal mesh is large)."""
    ctx = _context(template, lens_w, lens_h)
    before = {
        side: {
            "verts": np.asarray(ctx.mesh(f"{side}Rim").vertices).copy(),
            "faces": np.asarray(ctx.mesh(f"{side}Rim").faces).copy(),
            "outline": Polygon(LensDeformer._projected_footprint(ctx.mesh(f"{side}Rim"), f"{side}Rim").exterior),
            "opening": _opening(ctx, side.lower()),
        }
        for side in ("Left", "Right")
    }
    ctx = RimDeformer(contour_strength=0.8).apply(ctx, CONTOUR if with_contour else None)
    after = {
        side: {
            "verts": np.asarray(ctx.mesh(f"{side}Rim").vertices).copy(),
            "faces": np.asarray(ctx.mesh(f"{side}Rim").faces).copy(),
            "outline": Polygon(LensDeformer._projected_footprint(ctx.mesh(f"{side}Rim"), f"{side}Rim").exterior),
            "opening": _opening(ctx, side.lower()),
        }
        for side in ("Left", "Right")
    }
    return before, after, {r["side"]: r for r in ctx.metadata["rim_deformation"]["sides"]}


@pytest.mark.parametrize("template,lens_w,lens_h", [("geometric_metal", 46.0, 36.0), ("rectangle_plastic", 50.0, 40.0)])
@pytest.mark.parametrize("with_contour", [False, True])
def test_requested_size_reaches_the_rim_opening(template, lens_w, lens_h, with_contour):
    """A request that fits the template is achieved: opening = lens + clearance on every side."""
    before, after, meta = _run(template, lens_w, lens_h, with_contour)
    for side in ("Left", "Right"):
        w, h = _size(after[side]["opening"])
        assert w == pytest.approx(lens_w + 2 * CLEARANCE, abs=0.4), (side, w)
        assert h == pytest.approx(lens_h + 2 * CLEARANCE, abs=0.4), (side, h)
        assert not meta[side.lower()]["clamped_by_wall"]
        assert meta[side.lower()]["warning"] is None


def test_the_whole_rim_moves_not_a_single_vertex():
    before, after, meta = _run("geometric_metal", 46.0, 36.0, True)
    moved = np.linalg.norm(after["Left"]["verts"] - before["Left"]["verts"], axis=1) > 1e-6
    assert moved.sum() > 0.2 * len(moved), f"only {moved.sum()} of {len(moved)} vertices moved"
    assert meta["left"]["moved_vertices"] == int(moved.sum())


@pytest.mark.parametrize("template,lens_w,lens_h", [("geometric_metal", 46.0, 36.0), ("rectangle_plastic", 50.0, 40.0)])
def test_outer_outline_z_faces_are_untouched(template, lens_w, lens_h):
    before, after, _ = _run(template, lens_w, lens_h, True)
    for side in ("Left", "Right"):
        np.testing.assert_array_equal(after[side]["verts"][:, 2], before[side]["verts"][:, 2])   # Z
        np.testing.assert_array_equal(after[side]["faces"], before[side]["faces"])                # topology
        # exterior boundary only: the ring footprint also contains the opening, which SHOULD change
        drift = before[side]["outline"].symmetric_difference(after[side]["outline"]).area
        assert drift < 0.25, f"{side} rim outer outline moved ({drift:.3f} mm2): bridge/frame would detach"
        on_edge = np.array([before[side]["outline"].exterior.distance(Point(x, y)) < 0.02 for x, y in before[side]["verts"][:, :2]])
        moved = np.linalg.norm(after[side]["verts"][:, :2] - before[side]["verts"][:, :2], axis=1)
        assert moved[on_edge].max() < 0.02, f"{side}: a vertex ON the outer outline moved {moved[on_edge].max():.3f} mm"


def test_left_and_right_openings_are_symmetric():
    _, after, _ = _run("geometric_metal", 46.0, 36.0, True)
    lw, lh = _size(after["Left"]["opening"])
    rw, rh = _size(after["Right"]["opening"])
    assert lw == pytest.approx(rw, abs=0.3) and lh == pytest.approx(rh, abs=0.3)


def test_size_wins_over_photo_shape():
    """The photo outline is fuller in the corners than the template opening. It must not cost height."""
    _, with_c, meta_c = _run("geometric_metal", 46.0, 36.0, True)
    _, without, _ = _run("geometric_metal", 46.0, 36.0, False)
    assert _size(with_c["Left"]["opening"]) == pytest.approx(_size(without["Left"]["opening"]), abs=0.4)
    assert meta_c["left"]["contour_strength_used"] < meta_c["left"]["contour_strength_requested"]   # reported, not hidden


def test_impossible_request_is_clamped_and_reported():
    """geometric_metal's whole rim is ~58 mm wide: a 58 mm lens cannot fit, and that must be said."""
    _, after, meta = _run("geometric_metal", 58.0, 40.0, True)
    left = meta["left"]
    assert left["clamped_by_wall"] and left["warning"] and "no more room" in left["warning"]
    w, h = _size(after["Left"]["opening"])
    assert w < 58.0 + 2 * CLEARANCE - 3.0                       # genuinely smaller than asked
    assert w == pytest.approx(left["achieved_aperture_mm"][0], abs=0.4)   # and the report matches the mesh
    assert after["Left"]["opening"].within(after["Left"]["outline"])                              # still inside the rim
