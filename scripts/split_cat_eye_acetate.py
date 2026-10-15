#!/usr/bin/env python3
"""Build the ``cat_eye_acetate`` template from the raw ``cat_eye_glasses.glb``.

Raw layout (metres, Y up, front at +Z, temples toward -Z):
  Object_5 : the two lenses
  Object_7 : front frame layers + hinge bits + BOTH temples (4 thin shells per side)
  Object_9 : embossed logo fragments on the right temple (decorative, dropped)
Parts are identified by geometry, never by node name.

Usage: python scripts/split_cat_eye_acetate.py [source.glb] [templates_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_factory as tf  # noqa: E402

SPEC = tf.TemplateSpec(
    name="cat_eye_acetate", shape="cat_eye", material="acetate", style="acetate_cat_eye",
    bridge_type="keyhole", rim_pull_strength=0.45, temple_curve_angle=24.0, nose_pad_distance=17.0,
    source="cat_eye_glasses.glb", target_front_width=146.0, target_temple_length=140.0,
    notes="Upswept cat-eye acetate front; logo fragments from the source asset were dropped.",
)


def build(source: Path, out_dir: Path) -> None:
    comps = tf.load_components(source, scale=1000.0)
    lens = [c for c in comps if c.metadata["src"] == "Object_5"]
    logo = [c for c in comps if c.metadata["src"] == "Object_9"]
    rest = [c for c in comps if c.metadata["src"] == "Object_7"]
    if len(lens) != 2:
        raise ValueError(f"expected 2 lens components, found {len(lens)}")
    temples = [c for c in rest if c.extents[2] > 100]          # long in Z = temple shell
    others = [c for c in rest if c.extents[2] <= 100]
    front_parts = [c for c in others if c.extents[0] > 10 or c.centroid[2] > 100]
    hinge = [c for c in others if c.centroid[2] <= 100]
    print(f"lens={len(lens)} logo(dropped)={len(logo)} temple shells={len(temples)} front pieces={len(front_parts)} hinge bits={len(hinge)}")

    lens_by = {tf.side_of(m): m for m in lens}
    temple_by = {s: [m for m in temples + hinge if tf.side_of(m) == s] for s in ("Left", "Right")}
    front = tf.cat(front_parts)
    bridge_half, end_x, cut = tf.search_cut(front, lens_by)
    print(f"cut: bridge_half={bridge_half:.1f} end_x={end_x:.1f}")
    parts = {
        "Frame": cut["Frame"], "Bridge": cut["Bridge"], "LeftRim": cut["LeftRim"], "RightRim": cut["RightRim"],
        "LeftLens": lens_by["Left"], "RightLens": lens_by["Right"],
        "LeftTemple": tf.cat(temple_by["Left"]), "RightTemple": tf.cat(temple_by["Right"]),
    }
    tf.finish(SPEC, parts, out_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "templates" / "raw" / "cat_eye_glasses.glb"
    build(src, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "templates")
