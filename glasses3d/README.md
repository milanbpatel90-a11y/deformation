# Glasses3D

An image-to-GLB pipeline for virtual try-on. The core stages are independently importable and use deterministic OpenCV/procedural fallbacks when optional model packages or trained weights are unavailable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run from this directory:

```powershell
python main.py --front front.jpg --side side.jpg --frame-width-mm 140 --out model.glb --debug
```

The side image is optional. Without it, a warning is emitted and a 135 mm default temple is used. Debug mode writes a texture atlas and JSON report under `debug/`.

The generated scene has a root node with `front`, `temple_L`, and `temple_R` children. Geometry uses metres (1 unit = 1 m), and the template loader preserves named parts and hinge pivot metadata.

For production accuracy, replace the fallback segmenter/classifier/landmark hooks with trained rembg/SAM2 and keypoint models, and add calibrated templates under `templates/`.
