# 🚀 Execution Checklist - Automatic 3D Glasses Generation

Follow these steps in order to get your pipeline running.

---

## ✅ Phase 1: Setup & Verification (5 minutes)

### 1.1 Install Dependencies

```bash
pip install -r requirements.txt
```

**Verify:**
```bash
python quick_start_auto3d.py
```

Expected output: ✓ All dependencies installed

---

### 1.2 Verify YOLO

```bash
yolo checks
```

Expected: YOLO installation verified

---

### 1.3 Create Directories

```bash
python quick_start_auto3d.py
```

This creates:
- `dataset/images/train/`
- `dataset/images/val/`
- `dataset/labels/train/`
- `dataset/labels/val/`
- `output_3d/`

---

## ✅ Phase 2: Dataset Preparation (2-3 hours for 71 images)

### 2.1 Collect Images

**Option A: Use Unsplash Script**
```bash
python download_unsplash_glasses.py --count 71 --output dataset/images_raw
```

**Option B: Manual Collection**
- Download 71+ glasses product images
- Save to `dataset/images_raw/`
- Requirements:
  - Clear, front-facing view
  - Good lighting
  - Minimal background clutter
  - Resolution: 640px or higher

---

### 2.2 Annotate in CVAT

**Follow:** `CVAT_SETUP_AND_ANNOTATION.md`

1. **Setup CVAT:**
   ```bash
   docker-compose up -d
   ```
   Access: http://localhost:8080

2. **Create Project:**
   - Name: "Eyewear Segmentation"
   - Labels: 6 classes (rim, temple, bridge, left_lens, right_lens, nose_pad)

3. **Upload Images:**
   - Upload from `dataset/images_raw/`

4. **Annotate:**
   - Use polygon tool
   - Annotate all 6 parts for each image
   - Quality check: ensure accurate boundaries

5. **Export:**
   - Format: YOLO 1.1
   - Download to `dataset/`

---

### 2.3 Organize Dataset

```bash
# Move images
cp dataset/images_raw/*.jpg dataset/images/train/

# Move labels (from CVAT export)
cp cvat_export/obj_train_data/*.txt dataset/labels/train/

# Create validation split (10-15% of data)
# Move ~10 images and labels to val/ folders
```

**Verify structure:**
```
dataset/
├── images/
│   ├── train/  (60+ images)
│   └── val/    (10+ images)
└── labels/
    ├── train/  (60+ .txt files)
    └── val/    (10+ .txt files)
```

---

## ✅ Phase 3: Train YOLO Model (1-2 hours)

### 3.1 Validate Dataset

```bash
python train_yolov8_seg.py validate --dataset-root ./dataset
```

Expected: Dataset structure validated ✓

---

### 3.2 Start Training

**Quick Training (50 epochs):**
```bash
python train_yolov8_seg.py train \
  --dataset-yaml data.yaml \
  --epochs 50 \
  --batch-size 16 \
  --name eyewear_seg_v1
```

**Extended Training (100 epochs - Recommended):**
```bash
python train_yolov8_seg.py train \
  --dataset-yaml data.yaml \
  --epochs 100 \
  --batch-size 16 \
  --name eyewear_seg_v1
```

**Monitor:**
- Training progress in terminal
- Results: `runs/segment/eyewear_seg_v1/`

---

### 3.3 Check Results

```bash
# View training results
ls runs/segment/eyewear_seg_v1/

# Expected files:
# - weights/best.pt  ← Your trained model
# - results.png
# - confusion_matrix.png
```

**Target Metrics:**
- mAP50: > 0.85
- Segmentation mAP50: > 0.80

---

## ✅ Phase 4: Test Segmentation (10 minutes)

### 4.1 Test on Single Image

```bash
yolo task=segment mode=predict \
  model=runs/segment/eyewear_seg_v1/weights/best.pt \
  source=dataset/images/val/test_001.jpg \
  save=True
```

**Check output:**
```bash
# View prediction
open runs/segment/predict/test_001.jpg
```

---

### 4.2 Evaluate on Validation Set

```bash
python train_yolov8_seg.py evaluate \
  --model-path runs/segment/eyewear_seg_v1/weights/best.pt \
  --dataset-yaml data.yaml
```

---

## ✅ Phase 5: Generate 3D Models (5 minutes)

### 5.1 Test Single Image

```bash
python backend/pipeline/auto_3d_pipeline.py \
  dataset/images/val/test_001.jpg \
  -o output_3d \
  -m runs/segment/eyewear_seg_v1/weights/best.pt
```

**Check outputs:**
```bash
ls output_3d/
# Expected:
# - test_001.glb
# - test_001_measurements.json
# - test_001_segmentation.jpg
# - test_001.metadata.json
```

---

### 5.2 View 3D Model

**Option A: Online Viewer**
- Upload `output_3d/test_001.glb` to https://gltf-viewer.donmccurdy.com/

**Option B: Local Viewer**
```bash
# Open viewer/index.html in browser
# Drag and drop the GLB file
```

---

### 5.3 Batch Processing

```bash
python backend/pipeline/auto_3d_pipeline.py \
  dataset/images/val/ \
  -o output_3d_batch \
  -m runs/segment/eyewear_seg_v1/weights/best.pt \
  --batch
```

**Check summary:**
```bash
cat output_3d_batch/batch_summary.json
```

---

## ✅ Phase 6: Integration Testing (10 minutes)

### 6.1 Test with Custom Parameters

```bash
python backend/pipeline/auto_3d_pipeline.py \
  test_image.jpg \
  -o output \
  -m best.pt \
  --material metal \
  --color "#d4af37" \
  --reference-width 142
```

---

### 6.2 Python API Test

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

# Initialize
pipeline = Auto3DPipeline(
    yolo_model_path="runs/segment/eyewear_seg_v1/weights/best.pt"
)

# Process
result = pipeline.process_image(
    "test_image.jpg",
    "output"
)

print(f"Generated: {result['glb_model']}")
print(f"Measurements: {result['measurements_data']}")
```

---

## ✅ Phase 7: Production Deployment (Optional)

### 7.1 Create API Server

```python
# backend/api/main.py
from fastapi import FastAPI, UploadFile
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

app = FastAPI()
pipeline = Auto3DPipeline(yolo_model_path="best.pt")

@app.post("/generate-3d")
async def generate_3d(file: UploadFile):
    result = pipeline.process_image(file, "output")
    return result
```

**Run:**
```bash
uvicorn backend.api.main:app --reload
```

---

### 7.2 Test API

```bash
curl -X POST "http://localhost:8000/generate-3d" \
  -F "file=@glasses.jpg"
```

---

## 📊 Success Criteria

After completing all phases, you should have:

✅ Trained YOLO model with mAP50 > 0.85  
✅ Accurate segmentation of 6 glasses parts  
✅ Automatic measurement extraction working  
✅ 3D GLB models generated successfully  
✅ Models viewable in Three.js/GLB viewers  
✅ Batch processing functional  
✅ Complete documentation  

---

## 🐛 Troubleshooting

### Issue: Training fails with CUDA error
**Solution:**
```bash
# Use CPU
python train_yolov8_seg.py train --device cpu ...
```

### Issue: Poor segmentation quality
**Solution:**
- Check annotation quality
- Add more training data
- Increase epochs
- Use data augmentation

### Issue: 3D model looks incorrect
**Solution:**
- Check segmentation visualization
- Verify measurements.json
- Adjust reference_width_mm parameter

### Issue: Import errors
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

---

## 📈 Next Steps After Validation

Once you've validated with 71 images:

1. **Scale Up:**
   - Collect 200+ images
   - Retrain model
   - Improve accuracy

2. **Optimize:**
   - Fine-tune measurements
   - Improve 3D generation
   - Add more frame styles

3. **Deploy:**
   - Set up API server
   - Integrate with frontend
   - Add to production

4. **Monitor:**
   - Track accuracy metrics
   - Collect edge cases
   - Continuous improvement

---

## 🎉 You're Ready!

Follow this checklist step-by-step, and you'll have a working automatic 3D glasses generation pipeline.

**Estimated Total Time:**
- Setup: 5 minutes
- Dataset prep: 2-3 hours
- Training: 1-2 hours
- Testing: 15 minutes
- **Total: 4-6 hours**

**Start now with Phase 1! 🚀**

---

## 📞 Quick Reference

**Key Commands:**
```bash
# Setup
python quick_start_auto3d.py

# Train
python train_yolov8_seg.py train --dataset-yaml data.yaml --epochs 100

# Generate 3D
python backend/pipeline/auto_3d_pipeline.py image.jpg -o output -m best.pt

# Batch process
python backend/pipeline/auto_3d_pipeline.py ./images/ -o output -m best.pt --batch
```

**Key Files:**
- Model: `runs/segment/train/weights/best.pt`
- Config: `data.yaml`
- Pipeline: `backend/pipeline/auto_3d_pipeline.py`
- Docs: `AUTO_3D_PIPELINE_GUIDE.md`

---

**Good luck! 🎓👓**