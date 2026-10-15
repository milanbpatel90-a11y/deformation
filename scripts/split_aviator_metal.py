#!/usr/bin/env python3
"""Build the ``aviator_metal`` template from the raw ``glasses (12).glb``.

Raw layout: one node, 26 components, ~3.1x real size (a 448 mm wide frame), front at Z~0,
temples toward -Z.  Components are classified by geometry:

  lenses   : flat sheets (Z thickness < 1) wider than 100 mm
  rims     : wire rings wider than 100 mm and thicker than the lens
  bridge   : everything between the rims (top bar + nose-pad hardware), |centroid x| < 60
  frame    : the outboard end pieces / hinge barrels at |x| > 200 with short Z reach
  temples  : long rods / ear curves (Z reach > 100)

Usage: python scripts/split_aviator_metal.py [source.glb] [templates_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_factory as tf  # noqa: E402

SPEC = tf.TemplateSpec(
    name="aviator_metal", shape="aviator", material="metal", style="metal_aviator",
    bridge_type="straight", rim_pull_strength=0.40, temple_curve_angle=32.0, nose_pad_distance=19.0,
    source="glasses (12).glb", target_front_width=142.0, target_temple_length=140.0,
    notes="Wire-rim aviator, source was ~3.1x real size and is rescaled to eyewear size.",
)


def build(source: Path, out_dir: Path) -> None:
    comps = tf.load_components(source, scale=1.0)
    by = lambda pred: [c for c in comps if pred(c)]
    lenses = by(lambda c: c.extents[0] > 100 and c.extents[2] < 1.0)
    rims = by(lambda c: c.extents[0] > 100 and c.extents[2] >= 1.0)
    taken = {id(c) for c in lenses + rims}
    rest = [c for c in comps if id(c) not in taken]
    bridge = [c for c in rest if abs(c.centroid[0]) < 60]
    outer = [c for c in rest if abs(c.centroid[0]) >= 60]
    temples = [c for c in outer if c.extents[2] > 100]
    frame_pcs = [c for c in outer if c.extents[2] <= 100]
    if len(lenses) != 2 or len(rims) != 2:
        raise ValueError(f"expected 2 lenses + 2 rims, got {len(lenses)} + {len(rims)}")
    print(f"lenses=2 rims=2 bridge pieces={len(bridge)} frame(end pieces)={len(frame_pcs)} temple pieces={len(temples)}")

    pick = lambda group, s: tf.cat([c for c in group if tf.side_of(c) == s])
    parts = {
        "Frame": tf.cat(frame_pcs), "Bridge": tf.cat(bridge),
        "LeftRim": pick(rims, "Left"), "RightRim": pick(rims, "Right"),
        "LeftLens": pick(lenses, "Left"), "RightLens": pick(lenses, "Right"),
        "LeftTemple": pick(temples, "Left"), "RightTemple": pick(temples, "Right"),
    }
    tf.finish(SPEC, parts, out_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "templates" / "raw" / "glasses (12).glb"
    build(src, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "templates")
