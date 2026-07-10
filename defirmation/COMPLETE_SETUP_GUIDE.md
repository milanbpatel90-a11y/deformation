# 🚀 Complete Auto-3D Pipeline Setup Guide

## Overview

This guide will help you set up the complete automatic 3D glasses generation pipeline from scratch.

## Pipeline Flow

```
Product Image
     ↓
YOLO Segmentation (best.pt)
     ↓
Measurements Extraction
     ↓
Parametric GLB Generation
     ↓
3D Model Output
     ↓
Virtual Try-On (Three.js)
```

## Project Structure

```
AR_Glasses_AI/
│
├── backend/
│   ├── measurement/
│   │   ├── __init__.py
│   │   ├── mask_to_measurements.py    # Extract measurements from masks
│   │   └── extractor.py               # Fallback measurement extractor
│   │
│   ├── generator/
│   │   ├── __init__.py
│   │   └── parametric_glb_generator.py # Generate 3D GLB models
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── auto_3d_pipeline.py        # Main pipeline orchestrator
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py               # YOLO segmentation wrapper
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   └── shape_classifier.py        # Frame shape classification
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py                    # FastAPI endpoints
│
├── models/
│   └── best.pt                        # Trained YOLO model (to be created)
│
├── templates/
│   ├── round.glb                      # Template 3D models
│   ├── aviator.glb
│   ├── cat_eye.glb
│   └── rectangle.glb
│
├── output/                            # Generated 3D models
│
├── dataset/                           # Training data
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── test_images/                       # Test images for pipeline
```

## Step 1: Install Dependencies

### Required Packages

```bash
pip install -r requirements.txt
```

### Core Dependencies:
- **ultralytics**: YOLOv8 segmentation
- **trimesh**: 3D mesh processing
- **pygltflib**: GLB file generation
- **shapely**: 2D geometry operations
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **fastapi**: API framework
- **uvicorn**: ASGI server

## Step 2: Prepare Dataset

### 2.1 Collect Images

Use the provided script to download glasses images:

```bash
python download_unsplash_glasses.py
```

Or manually add images to `dataset/images/train/`

### 2.2 Annotate with CVAT

Follow the detailed guide in `CVAT_SETUP_AND_ANNOTATION.md`

**Classes to annotate:**
1. `rim` - Frame rim/border
2. `temple` - Temple arms
3. `bridge` - Bridge connecting lenses
4. `left_lens` - Left lens area
5. `right_lens` - Right lens area
6. `nose_pad` - Nose pads (if present)

### 2.3 Export Annotations

Export from CVAT in **YOLO 1.1 format** and place in `dataset/labels/train/`

## Step 3: Train YOLO Model

### 3.1 Verify data.yaml

Ensure `data.yaml` is configured correctly:

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 6
names:
  0: rim
  1: temple
  2: bridge
  3: left_lens
  4: right_lens
  5: nose_pad
```

### 3.2 Train the Model

```bash
python train_yolov8_seg.py train --dataset-yaml data.yaml --epochs 50 --batch 16
```

**Training Parameters:**
- Epochs: 50-100 (adjust based on dataset size)
- Batch size: 8-16 (adjust based on GPU memory)
- Image size: 640x640

### 3.3 Locate Trained Model

After training, the best model will be at:
```
runs/segment/train/weights/best.pt
```

Copy it to the models directory:
```bash
mkdir -p models
cp runs/segment/train/weights/best.pt models/best.pt
```

## Step 4: Create Template GLB Files

### Option A: Use Parametric Generator (Recommended)

The pipeline will automatically generate GLB files from measurements. No templates needed!

### Option B: Create Manual Templates

If you want to use template-based generation:

1. Create basic GLB models in Blender
2. Export as GLB format
3. Place in `templates/` directory:
   - `round.glb`
   - `aviator.glb`
   - `cat_eye.glb`
   - `rectangle.glb`

## Step 5: Test the Pipeline

### 5.1 Quick Test

```bash
python quick_start_auto3d.py --test
```

### 5.2 Process Single Image

```bash
python backend/pipeline/auto_3d_pipeline.py test_images/test_000_metal.jpg -o output -m models/best.pt
```

### 5.3 Batch Processing

```bash
python backend/pipeline/auto_3d_pipeline.py test_images/ -o output -m models/best.pt --batch
```

### 5.4 With Custom Parameters

```bash
python backend/pipeline/auto_3d_pipeline.py glasses.jpg \
  -o output \
  -m models/best.pt \
  --material metal \
  --color "#C0C0C0" \
  --reference-width 145.0
```

## Step 6: API Integration

### 6.1 Start API Server

```bash
uvicorn backend.api.main:app --reload --port 8000
```

### 6.2 API Endpoints

**Generate 3D Model:**
```bash
curl -X POST "http://localhost:8000/api/generate-3d" \
  -F "file=@glasses.jpg" \
  -F "material=metal" \
  -F "color=#000000"
```

**Response:**
```json
{
  "glb_url": "/output/glasses.glb",
  "measurements": {
    "frame_width": 140.5,
    "lens_width": 52.3,
    "lens_height": 45.2,
    "bridge_width": 18.5,
    "temple_length": 142.0
  },
  "shape": "geometric",
  "material": "metal"
}
```

## Step 7: Frontend Integration

### 7.1 Three.js Viewer

Use the provided viewer in `viewer/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/loaders/GLTFLoader.js"></script>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script src="viewer.js"></script>
</body>
</html>
```

### 7.2 Load Generated GLB

```javascript
const loader = new THREE.GLTFLoader();
loader.load('output/glasses.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

## Output Files

After processing, you'll get:

1. **GLB Model**: `output/glasses.glb`
   - 3D model ready for web viewing
   - Includes materials and textures

2. **Measurements JSON**: `output/glasses_measurements.json`
   ```json
   {
     "frame_width": 140.5,
     "lens_width": 52.3,
     "lens_height": 45.2,
     "bridge_width": 18.5,
     "temple_length": 142.0,
     "rim_thickness": 1.2,
     "material": "metal",
     "shape": "geometric",
     "color": "#000000"
   }
   ```

3. **Metadata**: `output/glasses.metadata.json`
   - Generator version
   - Processing timestamp
   - Full measurements

4. **Segmentation Visualization**: `output/glasses_segmentation.jpg`
   - Visual overlay of detected parts

## Troubleshooting

### Issue: "No YOLO model found"
**Solution:** Train the model first or provide path with `-m` flag

### Issue: "Failed to segment glasses"
**Solution:** 
- Ensure image has clear glasses view
- Check if model is trained on similar glasses
- Try adjusting image quality/lighting

### Issue: "Invalid GLB output"
**Solution:**
- Check trimesh installation
- Verify measurements are reasonable
- Check for geometry errors in logs

### Issue: "Poor segmentation quality"
**Solution:**
- Retrain with more diverse dataset
- Increase training epochs
- Add data augmentation

## Performance Optimization

### For Production:

1. **Use GPU**: Ensure CUDA is available for YOLO
2. **Batch Processing**: Process multiple images together
3. **Caching**: Cache generated models
4. **CDN**: Serve GLB files from CDN
5. **Compression**: Use Draco compression for GLB files

### Batch Processing Script:

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")
results = pipeline.process_batch(
    image_dir="input_images/",
    output_dir="output/",
    pattern="*.jpg"
)
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Collect and annotate dataset
3. ✅ Train YOLO model
4. ✅ Test pipeline with sample images
5. ✅ Integrate with API
6. ✅ Deploy to production

## Support

- **Documentation**: See `AUTO_3D_PIPELINE_GUIDE.md`
- **CVAT Setup**: See `CVAT_SETUP_AND_ANNOTATION.md`
- **Architecture**: See `ARCHITECTURE.md`

## License

MIT License - See LICENSE file for details

---

**Made with Bob** 🤖