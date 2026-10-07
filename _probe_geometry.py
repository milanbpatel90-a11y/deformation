"""Probe: do the new geometry features actually separate pose? (yaw is known truth)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.video.geometry import analyse_mask, estimate_yaw
from tests.multi_angle_fixture import orbit_views

rows = []
for yaw, _image, truth in orbit_views():
    geometry = analyse_mask(truth)
    rows.append((yaw, geometry))

fills = [g.bbox_fill for _y, g in rows if g]
fill_front, fill_side = max(fills), min(fills)
print(f"fill range on ground truth: {fill_side:.3f} .. {fill_front:.3f}\n")

print(f"{'yaw':>5s} {'asp':>6s} {'fill':>6s} {'temple':>7s} {'thinR':>6s} {'lensN':>5s} "
      f"{'clW':>5s} {'clH':>5s} {'tSpan':>6s} {'brSpan':>7s} {'skew':>7s} {'symm':>6s} {'yawEst':>7s}")
for yaw, g in rows:
    if g is None:
        print(f"{yaw:5.0f}   no geometry")
        continue
    yaw_est = estimate_yaw(g.bbox_fill, fill_front, fill_side)
    print(f"{yaw:5.0f} {g.aspect:6.2f} {g.bbox_fill:6.3f} {g.temple_visibility:7.3f} "
          f"{g.thin_column_ratio:6.3f} {g.lens_count:5d} {g.lens_cluster_width:5d} "
          f"{g.lens_cluster_height:5d} {g.temple_span:6d} {g.bridge_span:7d} "
          f"{g.lateral_skew:7.3f} {g.symmetry:6.3f} {yaw_est:7.1f}")

# Separability of the key cues: frontal (|yaw|<=15) vs perspective (25..50) vs profile (>=60).
print("\n--- feature ranges by pose band ---")
bands = {
    "front |yaw|<=15": [g for y, g in rows if g and abs(y) <= 15],
    "perspective 25-50": [g for y, g in rows if g and 25 <= abs(y) <= 50],
    "profile >=60": [g for y, g in rows if g and abs(y) >= 60],
}
for name, group in bands.items():
    if not group:
        continue
    tv = [g.temple_visibility for g in group]
    as_ = [g.aspect for g in group]
    fl = [g.bbox_fill for g in group]
    lc = [g.lens_count for g in group]
    print(f"{name:20s} temple {min(tv):.3f}-{max(tv):.3f}  aspect {min(as_):.2f}-{max(as_):.2f}  "
          f"fill {min(fl):.3f}-{max(fl):.3f}  lens_count {sorted(set(lc))}")

# Pitch views: does a plan view separate from a profile?
print("\n--- pitched (top) views ---")
for pitch in (30, 55, 75):
    for yaw in (0, 45):
        from tests.multi_angle_fixture import render_view
        _img, truth = render_view(yaw, pitch)
        g = analyse_mask(truth)
        if g:
            print(f"  yaw={yaw:3d} pitch={pitch:3d}  aspect {g.aspect:5.2f}  fill {g.bbox_fill:.3f}  "
                  f"temple {g.temple_visibility:.3f}  lensN {g.lens_count}  clH {g.lens_cluster_height}")
