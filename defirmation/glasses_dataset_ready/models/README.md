# 📦 YOLO Model Directory

## Place Your Trained Model Here

This directory should contain your trained YOLOv8 segmentation model.

### Expected File

```
models/
└── best.pt    # Your trained YOLO model
```

## How to Get the Model

### Option 1: Train Your Own Model

1. **Prepare Dataset**
   - Collect 100+ glasses images
   - Annotate in CVAT with 6 classes:
     - rim
     - temple
     - bridge
     - left_lens
     - right_lens
     - nose_pad

2. **Train YOLO**
   ```bash
   yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640
   ```

3. **Copy Trained Model**
   ```bash
   cp runs/segment/train/weights/best.pt models/best.pt
   ```

### Option 2: Use Pre-trained Model

If you have a pre-trained model:
```bash
cp /path/to/your/best.pt models/best.pt
```

## Model Requirements

- **Type**: YOLOv8 Segmentation (yolov8n-seg, yolov8s-seg, etc.)
- **Classes**: 6 (rim, temple, bridge, left_lens, right_lens, nose_pad)
- **Format**: PyTorch (.pt)
- **Input Size**: 640x640 (recommended)

## Using the Model

Once you have `best.pt` in this directory:

```python
from backend.pipeline.auto_3d_pipeline import Auto3DPipeline

# Initialize with your model
pipeline = Auto3DPipeline(yolo_model_path="models/best.pt")

# Process images
result = pipeline.process_image("glasses.jpg", "output")
```

## Without a Trained Model

The pipeline can work without a trained model using OpenCV fallback:

```python
# Initialize without model
pipeline = Auto3DPipeline()

# Will use OpenCV-based segmentation
result = pipeline.process_image("glasses.jpg", "output")
```

**Note**: Results will be better with a trained YOLO model.

## Model Performance

Expected performance with well-trained model:
- **mAP50**: > 0.85
- **mAP50-95**: > 0.65
- **Inference Time**: ~50-100ms per image (GPU)

## Troubleshooting

### Model Not Found
```
Error: No such file: models/best.pt
```
**Solution**: Place your trained model in this directory

### Wrong Model Type
```
Error: Model is not a segmentation model
```
**Solution**: Ensure you're using a YOLOv8-seg model, not detection

### Class Mismatch
```
Error: Expected 6 classes, got X
```
**Solution**: Retrain model with correct 6 classes

---

**Ready to add your model!** 🚀