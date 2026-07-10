# 🎯 Quick Start: Run the Auto-3D Pipeline

## Prerequisites Check

Before running, ensure you have:
- ✅ Python 3.8+ installed
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Trained YOLO model at `models/best.pt` (or will use fallback)
- ✅ Test images in `test_images/` directory

## Installation Status

Run the setup checker:
```bash
python quick_start_auto3d.py
```

If dependencies are missing:
```bash
pip install trimesh pygltflib shapely ultralytics opencv-python numpy fastapi uvicorn
```

## Usage Examples

### 1. Process Single Image (Basic)

```bash
python backend/pipeline/auto_3d_pipeline.py test_images/test_000_metal.jpg -o output
```

**Output:**
- `output/test_000_metal.glb` - 3D model
- `output/test_000_metal_measurements.json` - Measurements
- `output/test_000_metal_segmentation.jpg` - Visualization

### 2. Process with Trained Model

```bash
python backend/pipeline/auto_3d_pipeline.py test_images/test_000_metal.jpg \
  -o output \
  -m models/best.pt
```

### 3. Specify Material and Color

```bash
python backend/pipeline/auto_3d_pipeline.py glasses.jpg \
  -o output \
  -m models/best.pt \
  --material metal \
  --color "#C0C0C0"
```

**Available Materials:**
- `metal`
- `plastic`
- `acetate`
- `titanium`

### 4. Batch Processing

Process all images in a directory:

```bash
python backend/pipeline/auto_3d_pipeline.py test_images/ \
  -o output \
  -m models/best.pt \
  --batch
```

### 5. Custom Reference Width

If you know the actual frame width:

```bash
python backend/pipeline/auto_3d_pipeline.py glasses.jpg \
  -o output \
  -m models/best.pt \
  --reference-width 145.0
```

## Python API Usage

### Basic Usage

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

# Initialize pipeline
pipeline = Auto3DPipeline(
    yolo_model_path="models/best.pt",
    reference_width_mm=140.0
)

# Process single image
result = pipeline.process_image(
    image_path="test_images/test_000_metal.jpg",
    output_dir="output"
)

print(f"Generated: {result['glb_model']}")
print(f"Measurements: {result['measurements_data']}")
```

### Batch Processing

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

results = pipeline.process_batch(
    image_dir="test_images/",
    output_dir="output/",
    pattern="*.jpg"
)

for result in results:
    if "error" not in result:
        print(f"✓ {result['input_image']} -> {result['glb_model']}")
    else:
        print(f"✗ {result['input_image']}: {result['error']}")
```

### With Custom Parameters

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline
from backend.models import FrameMaterial

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="output",
    material=FrameMaterial.METAL,
    color="#FFD700"  # Gold color
)
```

## REST API Usage

### Start the API Server

```bash
uvicorn backend.api.main:app --reload --port 8000
```

### Generate 3D Model via API

```bash
curl -X POST "http://localhost:8000/api/generate-3d" \
  -F "file=@glasses.jpg" \
  -F "material=metal" \
  -F "color=#000000"
```

### Response Format

```json
{
  "success": true,
  "glb_url": "/output/glasses_20240625_123456.glb",
  "measurements": {
    "frame_width": 140.5,
    "lens_width": 52.3,
    "lens_height": 45.2,
    "bridge_width": 18.5,
    "temple_length": 142.0,
    "rim_thickness": 1.2,
    "material": "metal",
    "shape": "geometric",
    "color": "#000000"
  },
  "processing_time": 2.34
}
```

## Output Files Explained

### 1. GLB Model (`*.glb`)
- 3D model in GLB format
- Ready for Three.js/Babylon.js
- Includes materials and colors
- Optimized for web viewing

### 2. Measurements JSON (`*_measurements.json`)
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
  "nose_pads": true,
  "temple_curve_angle": 28.0,
  "color": "#000000"
}
```

### 3. Metadata (`*.metadata.json`)
```json
{
  "measurements": { ... },
  "generator": "ParametricGLBGenerator",
  "version": "1.0",
  "timestamp": "2024-06-25T12:34:56Z"
}
```

### 4. Segmentation Visualization (`*_segmentation.jpg`)
- Visual overlay showing detected parts
- Color-coded by class:
  - Green: Front view
  - Blue: Side view
  - Red: Full frame

## Viewing the 3D Model

### Option 1: Online Viewer
Upload to: https://gltf-viewer.donmccurdy.com/

### Option 2: Local Viewer
Open `viewer/index.html` in a browser

### Option 3: Blender
Import GLB file: File → Import → glTF 2.0

### Option 4: Three.js Code
```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

const loader = new GLTFLoader();
loader.load('output/glasses.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

## Common Issues & Solutions

### Issue: "No module named 'trimesh'"
```bash
pip install trimesh pygltflib shapely
```

### Issue: "YOLO model not found"
**Solution 1:** Train the model first
```bash
python train_yolov8_seg.py train --dataset-yaml data.yaml --epochs 50
```

**Solution 2:** Run without model (uses OpenCV fallback)
```bash
python backend/pipeline/auto_3d_pipeline.py glasses.jpg -o output
```

### Issue: "Failed to segment glasses"
- Ensure image shows glasses clearly
- Try different lighting/angle
- Check image quality (min 640x640)

### Issue: "Invalid measurements"
- Verify segmentation quality
- Adjust reference width: `--reference-width 145.0`
- Check if glasses are centered in image

## Performance Tips

### For Faster Processing:
1. Use GPU for YOLO (CUDA)
2. Reduce image size to 640x640
3. Use batch processing for multiple images
4. Cache generated models

### For Better Quality:
1. Use high-resolution input images
2. Ensure good lighting
3. Center glasses in frame
4. Use trained YOLO model
5. Provide accurate reference width

## Next Steps

1. **Train Your Model**: See `COMPLETE_SETUP_GUIDE.md` Step 3
2. **Integrate with Frontend**: See `COMPLETE_SETUP_GUIDE.md` Step 7
3. **Deploy API**: Use Docker or cloud platform
4. **Optimize Performance**: Enable GPU, caching, CDN

## Support

- Full documentation: `COMPLETE_SETUP_GUIDE.md`
- Pipeline details: `AUTO_3D_PIPELINE_GUIDE.md`
- CVAT annotation: `CVAT_SETUP_AND_ANNOTATION.md`

---

**Quick Reference:**
```bash
# Setup
python quick_start_auto3d.py

# Single image
python backend/pipeline/auto_3d_pipeline.py image.jpg -o output -m models/best.pt

# Batch
python backend/pipeline/auto_3d_pipeline.py images/ -o output -m models/best.pt --batch

# API
uvicorn backend.api.main:app --reload --port 8000
```

**Made with Bob** 🤖