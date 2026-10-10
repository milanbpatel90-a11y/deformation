"""Probe: does the real 1-class model detect the multi-angle fixture?

Also measures ground-truth silhouette geometry per yaw, which is what the view
classifier must be able to separate.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.segmentation.segmenter import GlassesSegmenter
from tests.multi_angle_fixture import orbit_views

seg = GlassesSegmenter()
print("model:", seg.model_type, "classes:", seg.class_count)

rows = []
tiles = []
for yaw, image, truth in orbit_views():
    masks = seg.segment(image)
    pred = masks["front"]
    inter = np.count_nonzero((pred > 0) & (truth > 0))
    union = np.count_nonzero((pred > 0) | (truth > 0))
    iou = inter / union if union else 0.0

    # Ground-truth silhouette geometry: what the classifier should latch onto.
    ys, xs = np.where(truth > 0)
    if len(xs) < 10:
        w = h = 0
        fill = aspect = 0.0
    else:
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        fill = float(len(xs)) / float(w * h)
        aspect = w / max(h, 1)
    rows.append((yaw, w, h, aspect, fill, masks["detections"], iou))
    tiles.append(cv2.resize(image, (213, 120)))

print(f"{'yaw':>5s} {'w':>5s} {'h':>5s} {'aspect':>7s} {'fill':>6s} {'det':>4s} {'IoU':>6s}")
for yaw, w, h, aspect, fill, det, iou in rows:
    print(f"{yaw:5.0f} {w:5d} {h:5d} {aspect:7.2f} {fill:6.3f} {det:4d} {iou:6.3f}")

widths = [r[1] for r in rows]
print(f"\nwidth range: {min(widths)} .. {max(widths)}  (ratio {min(widths)/max(widths):.2f})")
detected = sum(1 for r in rows if r[5] > 0)
print(f"frames with a detection: {detected}/{len(rows)}")
ious = [r[6] for r in rows if r[5] > 0]
if ious:
    print(f"IoU vs ground truth: min={min(ious):.3f} mean={sum(ious)/len(ious):.3f}")

# Contact sheet for visual inspection.
rows_of_tiles = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
sheet = np.vstack(rows_of_tiles)
out = Path("output/development/_multi_angle_probe")
out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / "orbit_contact_sheet.jpg"), sheet)
print("contact sheet:", out / "orbit_contact_sheet.jpg")

# A single frame at yaw 0 and 90 for close inspection.
for yaw in (0.0, 45.0, 90.0):
    image, truth = [v for v in orbit_views([yaw])][0][1:]
    cv2.imwrite(str(out / f"yaw_{int(yaw):03d}.jpg"), image)
    cv2.imwrite(str(out / f"yaw_{int(yaw):03d}_truth.png"), truth)
print("wrote single-pose frames to", out)
