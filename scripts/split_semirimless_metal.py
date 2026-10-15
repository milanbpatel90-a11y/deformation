#!/usr/bin/env python3
"""Build the ``browline_semirimless`` template from the raw ``glasses (1).glb``.

Raw asset: a clean 13-component semi-rimless metal frame in metres. The front is only a thin brow
wire + end pieces + bridge + pad arms, the lower half of each lens is held by an (unmodelled)
nylon line, so NO rim has a closed opening and the engine cannot fit a lens into it.

Fix, done explicitly rather than hidden: a thin closure wire (``CLOSURE_WIDTH`` mm wide,
``CLOSURE_DEPTH`` mm deep) is generated around each lens outline and merged into that side's rim, i.e. the
modelled nylon line. The brow wire stays the real geometry from the asset.

  lenses  : the two 7.5k-face sheets, wider than 40 mm at |x| in 8..60
  front   : the one component wider than 120 mm  -> cut into Bridge / Rims / Frame (end pieces)
  bridge  : nose-pad parts (|x| < 14 mm, not the front)
  temples : arm + hinge pin + rear piece per side

Usage: python scripts/split_semirimless_metal.py [source.glb] [templates_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_factory as tf  # noqa: E402
from backend.deformer.lens_deformer import LensDeformer  # noqa: E402

CLOSURE_WIDTH = 2.4
CLOSURE_DEPTH = 2.0

SPEC = tf.TemplateSpec(
    name="browline_semirimless", shape="browline", material="metal", style="metal_semirimless",
    bridge_type="straight", rim_pull_strength=0.40, temple_curve_angle=26.0, nose_pad_distance=18.0,
    source="glasses (1).glb", target_front_width=136.0, target_temple_length=140.0,
    notes="Semi-rimless: brow wire from the asset plus a generated closure wire along the open lower edge.",
)


def closure_ring(lens: trimesh.Trimesh) -> trimesh.Trimesh:
    outline = LensDeformer._projected_footprint(lens, "lens")
    poly = outline.buffer(CLOSURE_WIDTH + 0.05).difference(outline.buffer(0.05))
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    ring = trimesh.creation.extrude_polygon(poly, CLOSURE_DEPTH)
    zc = float(lens.vertices[:, 2].mean())
    ring.apply_translation([0, 0, zc - CLOSURE_DEPTH / 2.0])
    return ring


def build(source: Path, out_dir: Path) -> None:
    comps = tf.load_components(source, scale=1000.0, weld_seams=True)
    front = max(comps, key=lambda c: c.extents[0])
    if front.extents[0] < 120:
        raise ValueError("front component not found")
    rest = [c for c in comps if c is not front]
    lenses = [c for c in rest if len(c.faces) > 5000 and 40 < c.extents[0] < 60]
    lens_by = {tf.side_of(m): m for m in lenses}
    if set(lens_by) != {"Left", "Right"}:
        raise ValueError(f"expected one lens per side, got {sorted(lens_by)}")
    pads = [c for c in rest if abs(c.centroid[0]) < 14 and not any(c is l for l in lenses)]
    temples = [c for c in rest if not any(c is x for x in lenses + pads)]
    print(f"lenses=2 pad pieces={len(pads)} temple pieces={len(temples)}")

    rings = {s: closure_ring(lens_by[s]) for s in ("Left", "Right")}
    chosen = None
    for bridge_half in (9.0, 8.5, 9.5, 8.0, 10.0, 7.5, 10.5):
        cut = tf.cut_front(front, bridge_half, 61.0)
        rim = {s: tf.cat([cut[f"{s}Rim"], rings[s]]) for s in ("Left", "Right")}
        if all(tf.aperture_status(rim[s], lens_by[s], s)[0] for s in ("Left", "Right")):
            chosen = (bridge_half, cut, rim)
            break
    if chosen is None:
        raise ValueError("no bridge cut gives closed rim apertures")
    bridge_half, cut, rim = chosen
    print(f"cut: bridge_half={bridge_half}")

    pick = lambda group, s: tf.cat([c for c in group if tf.side_of(c) == s])
    parts = {
        "Frame": cut["Frame"], "Bridge": tf.cat([cut["Bridge"], *pads]),
        "LeftRim": rim["Left"], "RightRim": rim["Right"],
        "LeftLens": lens_by["Left"], "RightLens": lens_by["Right"],
        "LeftTemple": pick(temples, "Left"), "RightTemple": pick(temples, "Right"),
    }
    tf.finish(SPEC, parts, out_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "templates" / "raw" / "glasses (1).glb"
    build(src, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "templates")
