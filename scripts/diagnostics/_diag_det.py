"""Inspect the raw YOLO detections that inflate the mask at +10/+20 yaw."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.segmentation.segmenter import GlassesSegmenter
from tests.multi_angle_fixture import orbit_views

seg = GlassesSegmenter()
model = seg._model
print("model:", seg.model_type)

for yaw, image, truth in orbit_views():
    if yaw not in (10, 20, 0, -10, 30):
        continue
    results = model(image, verbose=False)
    print(f"\n=== yaw {yaw:+.0f}  truth cluster width "
          f"{int(np.count_nonzero(truth.any(axis=0)))}")
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("  no boxes")
            continue
        masks = result.masks.data if result.masks is not None else []
        for index in range(len(result.boxes)):
            conf = float(result.boxes.conf[index])
            cls = int(result.boxes.cls[index])
            x0, y0, x1, y1 = [float(v) for v in result.boxes.xyxy[index]]
            area = 0.0
            if index < len(masks):
                m = masks[index].cpu().numpy()
                area = float((m > 0.5).mean())
            print(f"  det {index}: conf={conf:.3f} cls={cls} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
                  f"w={x1-x0:.0f} mask_area_frac={area:.5f}")
