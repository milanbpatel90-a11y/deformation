# 🚀 Auto-3D Glasses Pipeline - Ready to Deploy

## 📦 What's Included

This directory contains the **complete, production-ready** auto-3D glasses generation pipeline.

## 📁 Directory Structure

```
glasses_dataset_ready/
│
├── backend/
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py              # YOLO segmentation wrapper
│   │
│   ├── measurement/
│   │   ├── __init__.py
│   │   └── mask_to_measurements.py   # Extract measurements from masks
│   │
│   ├── generator/
│   │   ├── __init__.py
│   │   └── parametric_glb_generator.py  # Generate 3D GLB models
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── auto_3d_pipeline.py       # Main pipeline orchestrator
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   └── shape_classifier.py       # Frame shape classification
│   │
│   ├── __init__.py
│   └── models.py                     # Data models (Measurements, etc.)
│
├── models/
│   └── best.pt                       # Place your trained YOLO model here
│
├── output/                           # Generated 3D models will be saved here
│
└── README.md                         # This file
```

## 🎯 Pipeline Flow

```
Product Image
     ↓
YOLO Segmentation (best.pt)
     ↓
Mask to Measurements
     ↓
Parametric GLB Generation
     ↓
3D Model Output (.glb)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics trimesh pygltflib shapely opencv-python numpy
```

### 2. Add Your YOLO Model

Place your trained model in the `models/` directory:
```bash
# Copy your trained model
cp /path/to/your/best.pt models/best.pt
```

### 3. Run the Pipeline

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

# Initialize pipeline
pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

# Process single image
result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="output"
)

print(f"Generated: {result['glb_model']}")
```

### 4. Command Line Usage

```bash
# Single image
python -m backend.pipeline.auto_3d_pipeline glasses.jpg -o output -m models/best.pt

# Batch processing
python -m backend.pipeline.auto_3d_pipeline images/ -o output -m models/best.pt --batch

# With custom parameters
python -m backend.pipeline.auto_3d_pipeline glasses.jpg \
  -o output \
  -m models/best.pt \
  --material metal \
  --color "#C0C0C0"
```

## 📊 Output Files

After processing, you'll get:

```
output/
├── glasses.glb                    # 3D model (ready for Three.js)
├── glasses_measurements.json      # Extracted measurements
├── glasses.metadata.json          # Generation metadata
└── glasses_segmentation.jpg       # Visualization
```

## 🔧 Core Components

### 1. Segmenter (`backend/segmentation/segmenter.py`)
- Wraps YOLO segmentation model
- Detects 6 classes: rim, temple, bridge, left_lens, right_lens, nose_pad
- Provides OpenCV fallback if no trained model

### 2. Mask to Measurements (`backend/measurement/mask_to_measurements.py`)
- Converts segmentation masks to parametric measurements
- Extracts: frame width, lens dimensions, bridge width, temple length
- Auto-calibration from image

### 3. GLB Generator (`backend/generator/parametric_glb_generator.py`)
- Generates 3D models from measurements
- Creates: lenses, frame rim, bridge, temples, nose pads
- Applies PBR materials with proper colors
- Exports to GLB format

### 4. Pipeline (`backend/pipeline/auto_3d_pipeline.py`)
- Orchestrates the complete workflow
- Handles single image and batch processing
- Auto-detects shape, material, and color
- Provides CLI interface

## 🎓 Training Your Model

If you don't have a trained model yet:

1. **Collect Images** (100+ recommended)
2. **Annotate in CVAT**
   - Classes: rim, temple, bridge, left_lens, right_lens, nose_pad
3. **Train YOLO**
   ```bash
   yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=50
   ```
4. **Copy Model**
   ```bash
   cp runs/segment/train/weights/best.pt models/best.pt
   ```

## 🌐 API Integration

### Python API

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline
from backend.models import FrameMaterial

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

# Process with custom parameters
result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="output",
    material=FrameMaterial.METAL,
    color="#FFD700"  # Gold
)
```

### Batch Processing

```python
results = pipeline.process_batch(
    image_dir="product_images/",
    output_dir="output/",
    pattern="*.jpg"
)

for result in results:
    if "error" not in result:
        print(f"✓ Generated: {result['glb_model']}")
```

## 🎨 Frontend Integration

### Three.js Example

```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

const loader = new GLTFLoader();
loader.load('output/glasses.glb', (gltf) => {
    scene.add(gltf.scene);
    
    // Position for virtual try-on
    gltf.scene.position.set(0, 0, -5);
    gltf.scene.rotation.y = Math.PI;
});
```

## 📋 Requirements

```
ultralytics>=8.1.0      # YOLO segmentation
trimesh>=4.2.0          # 3D mesh processing
pygltflib>=1.16.0       # GLB export
shapely>=2.0.0          # 2D geometry
opencv-python>=4.9.0    # Image processing
numpy>=1.26.0           # Numerical operations
```

## 🔍 Measurements Extracted

The pipeline extracts these measurements:

- **frame_width**: Total frame width (mm)
- **lens_width**: Individual lens width (mm)
- **lens_height**: Lens height (mm)
- **bridge_width**: Bridge width (mm)
- **temple_length**: Temple arm length (mm)
- **rim_thickness**: Frame rim thickness (mm)
- **temple_curve_angle**: Temple curve angle (degrees)
- **nose_pads**: Boolean (present/absent)
- **material**: metal, plastic, acetate, titanium
- **shape**: geometric, round, cat_eye, aviator, rimless
- **color**: Hex color code

## 🎯 Use Cases

1. **E-commerce**: Generate 3D models from product photos
2. **Virtual Try-On**: Create AR-ready glasses models
3. **Inventory Management**: Automated 3D catalog generation
4. **Custom Manufacturing**: Extract precise measurements
5. **Quality Control**: Verify frame dimensions

## 🚦 Status

- ✅ All modules implemented and tested
- ✅ Works with or without trained YOLO model
- ✅ Production-ready code
- ✅ Complete documentation
- ⏳ Add your trained model to `models/best.pt`

## 📞 Support

For issues or questions:
1. Check the code comments in each module
2. Review the measurement extraction logic
3. Verify YOLO model is properly trained
4. Test with sample images first

## 📄 License

MIT License - Free to use in commercial projects

---

**Ready to generate 3D glasses models! 🚀**

Place your trained `best.pt` in the `models/` directory and start processing images.