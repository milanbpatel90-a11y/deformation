# Defirmation Toolkit Guide

Full reference for template management, dataset prep, training, and calibration.

## Architecture

```
defirmation/
├── templates/
│   ├── geometric_metal.glb      # Base mesh (all stubs fallback here)
│   ├── geometric_metal.json     # Backend template spec
│   ├── templates.json           # Toolkit calibration registry
│   └── {stub}.json              # Per-style metadata
├── dataset/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/ # YOLO seg polygons
│   └── metadata/manifest.json
├── runs/segment/eyewear_seg/    # Trained weights
├── deformation_tuning_api.py    # Port 8001
└── backend/api/main.py          # Port 8000
```

## Part 1 — Templates

```powershell
python eyewear_vto_toolkit.py templates --create-stubs
python eyewear_vto_toolkit.py templates --list
```

Stubs created: `plastic_tortoise`, `acetate_black`, `cat_eye_gold`, `clubmaster_brown`, `oversized_clear`, `browline_metal`, `rimless_titanium`.

Each stub gets a `{name}.json` compatible with `backend.template_library`. Missing GLB files automatically fall back to `geometric_metal.glb`.

### Update calibration programmatically

```python
from eyewear_vto_toolkit import TemplateManager

tm = TemplateManager("./templates")
tm.update_calibration("plastic_tortoise", [8, 9, 10, 11, 12, 13, 14, 15], 0.68)
```

## Part 2 — Dataset Pipeline

1. Place images in `dataset/images/`
2. Place matching PNG masks in `dataset/masks/` (255 = eyewear)
3. Register entries:

```python
from eyewear_vto_toolkit import SegmentationDatasetPipeline

p = SegmentationDatasetPipeline("./dataset")
p.add_entry(
    "dataset/images/product_001.jpg",
    "dataset/masks/product_001.png",
    style="metal",
    bbox=(0.1, 0.1, 0.9, 0.9),
)
p.save_manifest()
```

4. Create splits + YOLO labels:

```powershell
python eyewear_vto_toolkit.py dataset --create-splits
python eyewear_vto_toolkit.py dataset --stats
```

## Part 3 — YOLOv8-Seg Training

```powershell
python train_yolov8_seg.py validate --dataset-root ./dataset
python train_yolov8_seg.py create-yaml --dataset-root ./dataset
python train_yolov8_seg.py train --dataset-yaml data.yaml --model yolov8n-seg --epochs 50
python train_yolov8_seg.py evaluate --model-path runs/segment/eyewear_seg/weights/best.pt --dataset-yaml data.yaml
python train_yolov8_seg.py export --model-path runs/segment/eyewear_seg/weights/best.pt
```

Set trained model for tuning API:

```powershell
$env:YOLO_MODEL_PATH = "runs/segment/eyewear_seg/weights/best.pt"
python deformation_tuning_api.py
```

## Part 4 — Deformation Tuning

1. `python deformation_tuning_api.py`
2. Open http://localhost:8001
3. Create stubs → select template → upload product photo → Detect Rim
4. Adjust rim pull slider → Confirm & Save

Saved values flow into:
- `templates/templates.json`
- `{template}.json` → `rim_pull_strength`
- `backend.deformer.MeshDeformer` via `TemplateLibrary.rim_pull_strength()`

### Interpreting metrics

| Metric | Good | Poor |
|--------|------|------|
| Edge confidence | >85% | <70% |
| Fit error | <15% | >30% |
| Rim pull | 0.5–0.7 typical | 0.1 min, 1.0 max |

## Part 5 — Main VTO Integration

```python
from backend.pipeline import DeformationPipeline

pipeline = DeformationPipeline()
result = pipeline.run_from_images("front.jpg", "side.jpg", "output/custom.glb")
```

The pipeline reads `rim_pull_strength` from template JSON automatically.

## Troubleshooting

**Template missing GLB** — Expected. System uses `geometric_metal.glb` with style-specific metadata.

**YOLO training slow** — Use `yolov8n-seg`, reduce `--batch-size`, or `--img-size 512`.

**Low rim confidence** — Train YOLO first, or use well-lit front product photos.

**Port conflict** — Tuning API = 8001, main API = 8000.
