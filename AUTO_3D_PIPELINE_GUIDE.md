# Automatic 3D Glasses Generation Pipeline

Complete guide for the end-to-end pipeline: **Product Image → 3D GLB Model**

---

## 🎯 Overview

This pipeline automatically generates 3D glasses models for virtual try-on from product images.

### Pipeline Flow

```
Product Image
     ↓
YOLO Segmentation (6 classes)
     ↓
Segmentation Masks
     ↓
Parametric Measurements
     ↓
3D GLB Generation
     ↓
Virtual Try-On Ready
```

---

## 📋 Prerequisites

### 1. Install Dependencies

```bash
pip install ultralytics trimesh pygltflib shapely opencv-python numpy
```

### 2. Verify Installation

```bash
yolo checks
python -c "import trimesh; print('Trimesh OK')"
```

---

## 🏗️ Phase 1: Dataset Preparation & Training

### Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── glasses_001.jpg
│   │   ├── glasses_002.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── glasses_001.txt
│   │   ├── glasses_002.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

### Label Format (YOLO Segmentation)

Each `.txt` file contains:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: 0-5 (rim, temple, bridge, left_lens, right_lens, nose_pad)
- `x, y`: Normalized polygon coordinates (0-1)

### Classes

```yaml
0: eyewear_rim      # Frame rim/front
1: eyewear_temple   # Temple arms
2: bridge           # Bridge connecting lenses
3: left_lens        # Left lens area
4: right_lens       # Right lens area
5: nose_pad         # Nose pads (if present)
```

### Train Model

```bash
# Basic training (50 epochs)
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16

# Extended training (100 epochs)
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  patience=20
```

### Training Output

```
runs/
└── segment/
    └── train/
        ├── weights/
        │   ├── best.pt          ← Use this
        │   └── last.pt
        ├── results.png
        ├── confusion_matrix.png
        └── val_batch0_pred.jpg
```

---

## 🔍 Phase 2: Test Segmentation

### Quick Test

```bash
yolo task=segment mode=predict \
  model=runs/segment/train/weights/best.pt \
  source=test_image.jpg \
  save=True
```

### Output

```
runs/segment/predict/
├── test_image.jpg          # Visualization
└── labels/
    └── test_image.txt      # Detected masks
```

---

## 🎨 Phase 3: Generate 3D Models

### Single Image

```bash
python backend/pipeline/auto_3d_pipeline.py \
  input_image.jpg \
  -o ./output \
  -m runs/segment/train/weights/best.pt
```

### Batch Processing

```bash
python backend/pipeline/auto_3d_pipeline.py \
  ./product_images/ \
  -o ./output_3d \
  -m runs/segment/train/weights/best.pt \
  --batch
```

### With Custom Parameters

```bash
python backend/pipeline/auto_3d_pipeline.py \
  glasses.jpg \
  -o ./output \
  -m best.pt \
  --material metal \
  --color "#d4af37" \
  --reference-width 142
```

---

## 📊 Output Files

For each input image, the pipeline generates:

```
output/
├── glasses_001.glb                    # 3D model
├── glasses_001.metadata.json          # Measurements
├── glasses_001_measurements.json      # Detailed measurements
├── glasses_001_segmentation.jpg       # Segmentation visualization
└── batch_summary.json                 # Batch processing summary
```

### GLB Model Structure

```
glasses.glb
├── left_lens          (transparent)
├── right_lens         (transparent)
├── frame_rim          (colored, material)
├── bridge             (colored, material)
├── left_temple        (colored, material)
├── right_temple       (colored, material)
├── left_nose_pad      (optional, soft)
└── right_nose_pad     (optional, soft)
```

---

## 🔧 Python API Usage

### Basic Usage

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

# Initialize pipeline
pipeline = Auto3DPipeline(
    yolo_model_path="runs/segment/train/weights/best.pt",
    reference_width_mm=140.0
)

# Process single image
result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="./output"
)

print(f"Generated: {result['glb_model']}")
print(f"Measurements: {result['measurements_data']}")
```

### Batch Processing

```python
# Process directory
results = pipeline.process_batch(
    image_dir="./product_images",
    output_dir="./output_3d",
    pattern="*.jpg"
)

print(f"Processed {len(results)} images")
```

### Custom Material & Color

```python
from backend.models import FrameMaterial

result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="./output",
    material=FrameMaterial.METAL,
    color="#d4af37"  # Gold
)
```

---

## 📐 Measurements Extracted

The pipeline automatically extracts:

```json
{
  "frame_width": 142.0,      // mm
  "lens_width": 52.0,        // mm
  "lens_height": 48.0,       // mm
  "bridge_width": 18.0,      // mm
  "temple_length": 145.0,    // mm
  "rim_thickness": 1.2,      // mm
  "temple_curve_angle": 28.0, // degrees
  "nose_pad_distance": 10.8, // mm (if present)
  "material": "metal",
  "shape": "geometric",
  "color": "#000000"
}
```

---

## 🌐 Integration with Virtual Try-On

### Load GLB in Three.js

```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('glasses.glb', (gltf) => {
  const glasses = gltf.scene;
  scene.add(glasses);
  
  // Position on face using MediaPipe landmarks
  positionGlassesOnFace(glasses, faceLandmarks);
});
```

### Position on Face

```javascript
function positionGlassesOnFace(glasses, landmarks) {
  // Get nose bridge position (landmark 168)
  const noseBridge = landmarks[168];
  
  glasses.position.set(
    noseBridge.x,
    noseBridge.y,
    noseBridge.z
  );
  
  // Scale based on face width
  const faceWidth = calculateFaceWidth(landmarks);
  const scale = faceWidth / 140; // 140mm reference
  glasses.scale.set(scale, scale, scale);
}
```

---

## 🚀 Production Deployment

### API Server

```python
from fastapi import FastAPI, UploadFile
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

app = FastAPI()
pipeline = Auto3DPipeline(yolo_model_path="best.pt")

@app.post("/generate-3d")
async def generate_3d(file: UploadFile):
    # Save uploaded image
    image_path = f"temp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Generate 3D model
    result = pipeline.process_image(image_path, "output")
    
    return {
        "glb_url": result["glb_model"],
        "measurements": result["measurements_data"]
    }
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download YOLO model
RUN yolo task=segment mode=train model=yolov8n-seg.pt epochs=0

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📈 Performance Optimization

### GPU Acceleration

```python
# Use GPU for YOLO inference
pipeline = Auto3DPipeline(
    yolo_model_path="best.pt"
)

# YOLO will automatically use CUDA if available
```

### Batch Processing

```python
# Process multiple images efficiently
results = pipeline.process_batch(
    image_dir="./images",
    output_dir="./output",
    pattern="*.jpg"
)
```

### Caching

```python
# Cache generated models
import hashlib

def get_cache_key(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

cache_key = get_cache_key("glasses.jpg")
cache_path = f"cache/{cache_key}.glb"

if os.path.exists(cache_path):
    return cache_path
else:
    result = pipeline.process_image(...)
    # Save to cache
```

---

## 🐛 Troubleshooting

### Issue: YOLO model not found

```bash
# Download pre-trained model
yolo task=segment mode=train model=yolov8n-seg.pt epochs=0
```

### Issue: Segmentation quality poor

- **Solution**: Train with more data (recommend 200+ images)
- Use data augmentation
- Increase training epochs

### Issue: 3D model looks incorrect

- **Solution**: Check segmentation masks first
- Adjust `reference_width_mm` parameter
- Verify lens contours are detected correctly

### Issue: Out of memory

- **Solution**: Reduce batch size
- Use smaller YOLO model (yolov8n-seg)
- Process images sequentially

---

## 📚 Advanced Usage

### Custom Lens Shapes

```python
from backend.models import LensContour

# Define custom lens contour
custom_contour = LensContour(
    left=[[0.1, 0.2], [0.3, 0.1], ...],
    right=[[0.7, 0.1], [0.9, 0.2], ...]
)

# Generate with custom contour
glb_path = pipeline.glb_generator.generate_from_measurements(
    measurements, custom_contour, "output.glb"
)
```

### Material Customization

```python
# Custom PBR materials
measurements.material = FrameMaterial.METAL
measurements.color = "#d4af37"  # Gold

# Generate with custom material
result = pipeline.process_image(...)
```

---

## 🎓 Training Tips

### Minimum Dataset Size

- **Validation**: 71 images (as mentioned)
- **Production**: 200+ images recommended
- **High Quality**: 500+ images

### Annotation Guidelines

1. **Rim**: Outline the entire frame front
2. **Temple**: Trace both temple arms
3. **Bridge**: Mark the bridge area precisely
4. **Lenses**: Separate left and right lens areas
5. **Nose Pads**: Mark if visible

### Data Augmentation

```yaml
# In training
augment: True
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10
translate: 0.1
scale: 0.5
flipud: 0.0
fliplr: 0.5
```

---

## 📞 Support

For issues or questions:
1. Check segmentation visualization first
2. Verify measurements.json output
3. Test with different reference widths
4. Review training metrics

---

## 🎉 Success Metrics

A well-trained model should achieve:

- **mAP50**: > 0.85
- **Segmentation mAP50**: > 0.80
- **Processing time**: < 2 seconds per image
- **3D model accuracy**: ±2mm from actual measurements

---

## 🔄 Continuous Improvement

1. Collect failed cases
2. Add to training dataset
3. Retrain model
4. Deploy updated model
5. Monitor performance

---

**Ready to generate 3D glasses automatically! 🚀**