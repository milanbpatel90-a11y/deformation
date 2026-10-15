#!/usr/bin/env python3
"""Build the ``round_metal`` template from the raw ``glasses (3).glb``.

Raw asset: thin-wire round frame, authored ~10x too small, lying on its side (front plane is YZ,
temples run along +X), seven nodes named a_A06 ... a_I05 (DAZ export names, meaningless here).
It is rotated 90 deg about Y so the front faces +Z / temples run to -Z, scaled x10 (cm -> mm) and then
split by GEOMETRY:

  lenses  : the thin a_B05 discs (two sheets per lens), centred on each eye
  rims    : every other component sitting in the front slab with |x| >= 13 mm (wire rings, rim lips)
  bridge  : components within |x| < 13 mm of the nose (bridge arch + nose-pad arms and pads)
  frame   : the end pieces / hinge blocks (a_A06 pieces that stay within |x| <= 79 mm)
  temples : remaining a_A06 / a_E08 pieces (hinge arm + temple + ear hook), split by side

Usage: python scripts/split_round_metal.py [source.glb] [templates_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_factory as tf  # noqa: E402

SPEC = tf.TemplateSpec(
    name="round_metal", shape="round", material="metal", style="metal_round",
    bridge_type="keyhole", rim_pull_strength=0.40, temple_curve_angle=30.0, nose_pad_distance=17.0,
    source="glasses (3).glb", target_front_width=138.0, target_temple_length=140.0,
    notes="Thin-wire round frame with pad arms; source was ~10x too small and on its side.",
)

BRIDGE_HALF_X = 13.0     # mm, pre-fit; components centred nearer the nose than this are bridge hardware
END_PIECE_MAX_X = 79.0   # mm, pre-fit; a_A06 pieces that stay inside this are end pieces, not temple


def build(source: Path, out_dir: Path) -> None:
    comps = tf.load_components(source, scale=10.0, transform=tf.rotation_y(90.0), weld_seams=True)
    name = lambda c: c.metadata["geom"]
    lens = [c for c in comps if name(c) == "a_B05"]
    others = [c for c in comps if name(c) != "a_B05"]
    cx = lambda c: abs(float(c.centroid[0]))
    temple_src = [c for c in others if name(c) in ("a_A06", "a_E08")]
    front_src = [c for c in others if name(c) not in ("a_A06", "a_E08")]
    frame = [c for c in temple_src if name(c) == "a_A06" and abs(c.bounds[:, 0]).max() <= END_PIECE_MAX_X]
    temples = [c for c in temple_src if not any(c is f for f in frame)]
    bridge = [c for c in front_src if cx(c) < BRIDGE_HALF_X]
    rims = [c for c in front_src if cx(c) >= BRIDGE_HALF_X]
    print(f"lens sheets={len(lens)} rim pieces={len(rims)} bridge pieces={len(bridge)} frame pieces={len(frame)} temple pieces={len(temples)}")

    pick = lambda group, s: tf.cat([c for c in group if tf.side_of(c) == s])
    parts = {
        "Frame": tf.cat(frame), "Bridge": tf.cat(bridge),
        "LeftRim": pick(rims, "Left"), "RightRim": pick(rims, "Right"),
        "LeftLens": pick(lens, "Left"), "RightLens": pick(lens, "Right"),
        "LeftTemple": pick(temples, "Left"), "RightTemple": pick(temples, "Right"),
    }
    tf.finish(SPEC, parts, out_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "templates" / "raw" / "glasses (3).glb"
    build(src, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "templates")
