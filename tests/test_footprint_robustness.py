"""Footprint union must survive meshes that make GEOS throw 'Ring edge missing'.

tests/data/rim_footprint_topology_failure.npz is the real RightRim of geometric_metal after the rim
stage reshaped it for the photo set of a real run. A plain shapely union_all() raises
TopologyException on it; before the fallback this crashed the whole pipeline with a 500.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.errors import GEOSException

from backend.deformer import lens_deformer
from backend.deformer.lens_deformer import LensDeformer

FIXTURE = Path(__file__).parent / "data" / "rim_footprint_topology_failure.npz"


def _mesh() -> trimesh.Trimesh:
    data = np.load(FIXTURE)
    return trimesh.Trimesh(vertices=data["v"], faces=data["f"].astype(np.int64), process=False)


def test_fixture_still_defeats_a_plain_union():
    """Guard the guard: if shapely/GEOS get better, this fixture stops testing anything."""
    mesh = _mesh()
    tri = mesh.vertices[mesh.faces][:, :, :2]
    polys = lens_deformer.polygons(lens_deformer.linearrings(tri))
    polys = polys[lens_deformer.shapely_area(polys) > 1e-7]
    try:
        lens_deformer.union_all(polys)
    except GEOSException:
        return
    pytest.skip("this GEOS build no longer fails on the fixture; the fallback test below covers the mechanism")


def test_real_failing_rim_projects_to_a_sane_footprint():
    footprint = LensDeformer._projected_footprint(_mesh(), "RightRim")
    assert footprint.is_valid and footprint.area == pytest.approx(267.34, abs=0.2)
    x0, y0, x1, y1 = footprint.bounds
    assert (x0, y0, x1, y1) == pytest.approx((10.07, 1.66, 67.96, 44.41), abs=0.05)


def test_fallback_grids_are_tried_in_order_and_failure_is_a_clear_error(monkeypatch):
    seen: list = []

    def always_fails(geoms, grid_size=None):
        seen.append(grid_size)
        raise GEOSException("TopologyException: Ring edge missing")

    monkeypatch.setattr(lens_deformer, "union_all", always_fails)
    with pytest.raises(ValueError, match="could not be merged into a footprint"):
        LensDeformer._projected_footprint(_mesh(), "RightRim")
    assert seen == [None, 1e-6, 1e-4, 1e-3]
