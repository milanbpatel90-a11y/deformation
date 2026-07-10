# Defirmation VTO — Quick Start

## 5-Minute Setup

```powershell
cd defirmation
pip install -r requirements.txt
python bootstrap.py
```

This creates directories, `templates/templates.json`, 7 template stubs, and `data.yaml`.

## Four Modules

| Module | File | Purpose |
|--------|------|---------|
| Template Manager | `eyewear_vto_toolkit.py` | Stubs, metadata, fallback chain |
| Data Pipeline | `eyewear_vto_toolkit.py dataset` | Manifest + YOLO splits |
| Training | `train_yolov8_seg.py` | YOLOv8-Seg train/eval/export |
| Tuning UI | `deformation_tuning_api.py` | Rim pull calibration (port **8001**) |

Main VTO API runs on port **8000** (`uvicorn backend.api.main:app`).

## Immediate Commands

```powershell
# List templates
python eyewear_vto_toolkit.py templates --list

# Deform from measurements (no images)
python run.py measurements -i examples/rose_gold_geometric.json -o output/test.glb

# Start tuning UI
python deformation_tuning_api.py
# → http://localhost:8001

# Start main API
uvicorn backend.api.main:app --reload --port 8000
```

## Week Roadmap

- **Week 2:** `python bootstrap.py` → 7 stubs ✓
- **Week 3:** Label 200+ images → `dataset/images/` + `dataset/masks/`
- **Week 4:** `python train_yolov8_seg.py train --dataset-yaml data.yaml`
- **Week 5:** Calibrate `rim_pull_strength` per style via tuning UI
- **Week 6:** Full VTO integration

See `TOOLKIT_GUIDE.md` for the full walkthrough.
