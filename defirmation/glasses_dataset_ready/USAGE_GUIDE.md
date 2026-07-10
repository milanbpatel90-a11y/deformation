# 📖 Usage Guide - Auto-3D Glasses Pipeline

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Model
```bash
# Copy your trained YOLO model
cp /path/to/your/best.pt models/best.pt
```

### Step 3: Generate 3D Model
```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")
result = pipeline.process_image("glasses.jpg", "output")
print(f"Generated: {result['glb_model']}")
```

## Command Line Usage

### Single Image Processing

```bash
# Basic usage
python -m backend.pipeline.auto_3d_pipeline glasses.jpg -o output -m models/best.pt

# With custom parameters
python -m backend.pipeline.auto_3d_pipeline glasses.jpg \
  -o output \
  -m models/best.pt \
  --material metal \
  --color "#C0C0C0" \
  --reference-width 145.0
```

### Batch Processing

```bash
# Process all images in a directory
python -m backend.pipeline.auto_3d_pipeline images/ \
  -o output \
  -m models/best.pt \
  --batch
```

### Available Options

```
-o, --output DIR          Output directory (default: ./output_3d)
-m, --model PATH          Path to YOLO model (optional)
--material TYPE           Frame material: metal, plastic, acetate, titanium
--color HEX               Frame color in hex (e.g., #000000)
--reference-width MM      Reference frame width for calibration (default: 140.0)
--batch                   Process directory in batch mode
```

## Python API

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
    image_path="glasses.jpg",
    output_dir="output"
)

# Access results
print(f"GLB Model: {result['glb_model']}")
print(f"Measurements: {result['measurements_data']}")
print(f"Shape: {result['shape']}")
print(f"Material: {result['material']}")
```

### With Custom Parameters

```python
from backend.models import FrameMaterial

result = pipeline.process_image(
    image_path="glasses.jpg",
    output_dir="output",
    material=FrameMaterial.METAL,
    color="#FFD700"  # Gold color
)
```

### Batch Processing

```python
results = pipeline.process_batch(
    image_dir="product_images/",
    output_dir="output/",
    pattern="*.jpg"
)

# Process results
for result in results:
    if "error" not in result:
        print(f"✓ {result['input_image']} -> {result['glb_model']}")
    else:
        print(f"✗ {result['input_image']}: {result['error']}")
```

### Without Trained Model

```python
# Use OpenCV fallback (no YOLO model required)
pipeline = Auto3DPipeline()  # No model path

result = pipeline.process_image("glasses.jpg", "output")
```

## Output Files

After processing, you'll get 4 files:

### 1. GLB Model (`*.glb`)
3D model ready for web viewing
```javascript
// Load in Three.js
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
const loader = new GLTFLoader();
loader.load('output/glasses.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

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
Visual overlay showing detected parts

## Advanced Usage

### Custom Measurement Extraction

```python
from backend.measurement.mask_to_measurements import MaskToMeasurements
from backend.models import FrameShape, FrameMaterial

converter = MaskToMeasurements(reference_width_mm=145.0)

# Extract from YOLO results
measurements, lens_contour = converter.extract_from_yolo_results(
    image=image,
    results=yolo_results,
    shape=FrameShape.GEOMETRIC,
    material=FrameMaterial.METAL,
    color="#000000"
)
```

### Direct GLB Generation

```python
from backend.generator.parametric_glb_generator import ParametricGLBGenerator

generator = ParametricGLBGenerator()

glb_path = generator.generate_from_measurements(
    measurements=measurements,
    lens_contour=lens_contour,
    output_path="output/glasses.glb"
)
```

### Custom Segmentation

```python
from backend.segmentation.segmenter import GlassesSegmenter

segmenter = GlassesSegmenter(model_path="models/best.pt")
masks = segmenter.segment(image)

# Access individual masks
rim_mask = masks.get("rim")
lens_mask = masks.get("left_lens")
```

## Material Options

```python
from backend.models import FrameMaterial

# Available materials
FrameMaterial.METAL      # Metallic finish
FrameMaterial.PLASTIC    # Matte plastic
FrameMaterial.ACETATE    # Glossy acetate
FrameMaterial.TITANIUM   # Brushed titanium
```

## Shape Options

```python
from backend.models import FrameShape

# Available shapes (auto-detected)
FrameShape.GEOMETRIC     # Angular frames
FrameShape.ROUND         # Circular frames
FrameShape.CAT_EYE       # Cat-eye style
FrameShape.AVIATOR       # Aviator style
FrameShape.RIMLESS       # Rimless frames
FrameShape.BROWLINE      # Browline/clubmaster
FrameShape.OVERSIZED     # Oversized frames
```

## Error Handling

```python
try:
    result = pipeline.process_image("glasses.jpg", "output")
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Processing failed: {e}")
```

## Performance Tips

### For Faster Processing
1. Use GPU for YOLO inference
2. Reduce image size to 640x640
3. Use batch processing for multiple images
4. Cache generated models

### For Better Quality
1. Use high-resolution input images (min 640x640)
2. Ensure good lighting and contrast
3. Center glasses in frame
4. Use trained YOLO model
5. Provide accurate reference width

## Viewing 3D Models

### Option 1: Online Viewer
Upload to: https://gltf-viewer.donmccurdy.com/

### Option 2: Three.js
```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();

const loader = new GLTFLoader();
loader.load('output/glasses.glb', (gltf) => {
    scene.add(gltf.scene);
    camera.position.z = 5;
    
    function animate() {
        requestAnimationFrame(animate);
        gltf.scene.rotation.y += 0.01;
        renderer.render(scene, camera);
    }
    animate();
});
```

### Option 3: Blender
1. Open Blender
2. File → Import → glTF 2.0
3. Select your .glb file

## Troubleshooting

### Issue: "No module named 'trimesh'"
```bash
pip install trimesh pygltflib shapely
```

### Issue: "YOLO model not found"
**Solution 1**: Add model to `models/best.pt`
**Solution 2**: Run without model (uses OpenCV fallback)
```python
pipeline = Auto3DPipeline()  # No model path
```

### Issue: "Failed to segment glasses"
- Ensure image shows glasses clearly
- Check image quality (min 640x640)
- Try different lighting/angle
- Verify model is trained on similar glasses

### Issue: "Invalid measurements"
- Verify segmentation quality
- Adjust reference width: `--reference-width 145.0`
- Check if glasses are centered in image
- Review segmentation visualization

## Examples

### E-commerce Integration
```python
# Process product catalog
import os
from pathlib import Path

pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

product_dir = Path("product_images")
for image_path in product_dir.glob("*.jpg"):
    try:
        result = pipeline.process_image(image_path, "output")
        print(f"✓ Processed: {image_path.name}")
        
        # Upload to CDN
        # upload_to_cdn(result['glb_model'])
    except Exception as e:
        print(f"✗ Failed: {image_path.name} - {e}")
```

### Virtual Try-On
```python
# Generate AR-ready models
result = pipeline.process_image(
    "glasses.jpg",
    "output",
    material=FrameMaterial.METAL,
    color="#FFD700"
)

# Use in AR application
ar_model_path = result['glb_model']
measurements = result['measurements_data']
```

### Quality Control
```python
# Extract and verify measurements
result = pipeline.process_image("glasses.jpg", "output")
measurements = result['measurements_data']

# Check tolerances
if abs(measurements['frame_width'] - 140.0) > 2.0:
    print("⚠️ Frame width out of tolerance")
if abs(measurements['bridge_width'] - 18.0) > 1.0:
    print("⚠️ Bridge width out of tolerance")
```

---

**Ready to generate 3D models!** 🚀

For more details, see the main README.md