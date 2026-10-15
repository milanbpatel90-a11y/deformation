#!/usr/bin/env python3
"""Build the ``rectangle_acetate`` template from the raw ``glasses (5).glb``.

The source is a lens-less, full-rim rectangular acetate frame that is

  * yawed ~19.8 deg about Y (the front runs diagonally across X/Z),
  * extremely dense (276k faces, 2,456 loose components of screws/hinge detail),
  * delivered as three fused nodes (the right temple is welded to the front).

So: un-yaw, quadric-decimate the whole assembly (keeps the silhouette and the lens openings), split
front / temples by depth, synthesize the missing lenses from the rim openings, then let the
engine's own aperture check choose the bridge / end-piece cuts.

Usage: python scripts/split_rectangle_acetate.py [source.glb] [templates_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_factory as tf  # noqa: E402

YAW_DEG = 19.8          # un-yaw so the front runs along X
TARGET_REDUCTION = 0.9  # 276k -> ~28k faces
FRONT_Z_MIN = 44.0      # faces with centroid Z at/above this belong to the front (temples hang behind)

SPEC = tf.TemplateSpec(
    name="rectangle_acetate", shape="rectangle", material="acetate", style="acetate_rectangle",
    bridge_type="keyhole", rim_pull_strength=0.45, temple_curve_angle=22.0, nose_pad_distance=18.0,
    source="glasses (5).glb", target_front_width=138.0, target_temple_length=140.0,
    notes="Thick rectangular acetate; decimated from 276k faces, lenses generated from the rim openings.",
)


def build(source: Path, out_dir: Path) -> None:
    import fast_simplification as fs

    comps = tf.load_components(source, scale=1.0, transform=tf.rotation_y(YAW_DEG))
    whole = tf.cat(comps)
    v, f = fs.simplify(np.asarray(whole.vertices, dtype=np.float32), np.asarray(whole.faces, dtype=np.int32),
                       target_reduction=TARGET_REDUCTION)
    whole = trimesh.Trimesh(v.astype(np.float64), f, process=False)
    whole.merge_vertices()
    print(f"decimated to {len(whole.faces)} faces")

    cz = whole.triangles_center
    cx = whole.triangles_center[:, 0]
    mid = 0.5 * (whole.bounds[0][0] + whole.bounds[1][0])
    front_mask = cz[:, 2] >= FRONT_Z_MIN
    front = tf.sub(whole, front_mask)
    rest = ~front_mask
    temples = {"Left": tf.sub(whole, rest & (cx < mid)), "Right": tf.sub(whole, rest & (cx >= mid))}
    # centre on X before cutting so Left = -X and the cuts are symmetric
    dx = -0.5 * (front.bounds[0][0] + front.bounds[1][0])
    for m in (front, temples["Left"], temples["Right"]):
        m.apply_translation([dx, 0, 0])

    lens = tf.synthesize_lenses(front)
    bridge_half, end_x, cut = tf.search_cut(front, lens)
    print(f"cut: bridge_half={bridge_half:.1f} end_x={end_x:.1f}")
    parts = {
        "Frame": cut["Frame"], "Bridge": cut["Bridge"], "LeftRim": cut["LeftRim"], "RightRim": cut["RightRim"],
        "LeftLens": lens["Left"], "RightLens": lens["Right"],
        "LeftTemple": temples["Left"], "RightTemple": temples["Right"],
    }
    tf.finish(SPEC, parts, out_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "templates" / "raw" / "glasses (5).glb"
    build(src, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "templates")
