"""Compare segmented masks with the ground-truth silhouettes, per orbit pose."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.segmentation.segmenter import GlassesSegmenter
from backend.video.geometry import analyse_mask
from tests.multi_angle_fixture import orbit_views

seg = GlassesSegmenter()
print(f"{'yaw':>5s} {'truth clW':>10s} {'seg clW':>8s} {'truth clH':>10s} {'seg clH':>8s} "
      f"{'det':>4s} {'IoU':>6s} {'seg aspect':>10s} {'seg fill':>9s}")
rows = []
for yaw, image, truth in orbit_views():
    masks = seg.segment(image)
    pred = masks["front"]
    inter = np.count_nonzero((pred > 0) & (truth > 0))
    union = np.count_nonzero((pred > 0) | (truth > 0))
    iou = inter / union if union else 0.0
    gt = analyse_mask(truth)
    sg = analyse_mask(pred)
    rows.append((yaw, gt, sg, masks["detections"], iou, masks.get("fallback"), masks["model_type"]))
    print(f"{yaw:5.0f} {gt.lens_cluster_width if gt else 0:10d} "
          f"{sg.lens_cluster_width if sg else 0:8d} "
          f"{gt.lens_cluster_height if gt else 0:10d} "
          f"{sg.lens_cluster_height if sg else 0:8d} "
          f"{masks['detections']:4d} {iou:6.3f} "
          f"{sg.aspect if sg else 0:10.2f} {sg.bbox_fill if sg else 0:9.3f}")

print("\n--- mask sources ---")
for yaw, _gt, _sg, det, _iou, fallback, model in rows:
    if yaw in (-90, -60, -20, 0, 20, 60, 90):
        print(f"  yaw {yaw:4.0f}: {model} detections={det} fallback={fallback}")

print("\n--- where does a wide cluster come from? ---")
for yaw, _gt, sg, _det, _iou, _fb, _m in rows:
    if sg is not None and sg.lens_cluster_width > 500:
        print(f"  yaw {yaw:4.0f}: seg cluster {sg.lens_cluster_width}px  runs={sg.lens_runs}  "
              f"temple_vis={sg.temple_visibility:.3f}  clH={sg.lens_cluster_height}")
