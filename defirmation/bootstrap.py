#!/usr/bin/env python3
"""Bootstrap the Defirmation VTO toolkit — run once to initialize."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def check_dependencies() -> bool:
    deps = {
        "fastapi": "FastAPI",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "trimesh": "trimesh",
        "pydantic": "Pydantic",
    }
    optional = {"ultralytics": "YOLOv8 (optional until training)", "torch": "PyTorch (optional)"}
    missing = []
    for pkg, desc in deps.items():
        try:
            __import__(pkg)
            logger.info("OK  %-15s %s", pkg, desc)
        except ImportError:
            logger.warning("MISS %-15s %s", pkg, desc)
            missing.append(pkg)
    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            logger.info("OK  %-15s %s", pkg, desc)
        except ImportError:
            logger.info("SKIP %-15s %s", pkg, desc)
    if missing:
        logger.error("Install missing: pip install -r requirements.txt")
        return False
    return True


def create_directories() -> None:
    dirs = [
        "templates",
        "dataset/images",
        "dataset/masks",
        "dataset/metadata",
        "dataset/splits/train",
        "dataset/splits/val",
        "dataset/splits/test",
        "dataset/labels",
        "runs/segment",
        "output",
        "toolkit",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        logger.info("OK  %s/", d)


def init_templates() -> None:
    from eyewear_vto_toolkit import TemplateManager

    tm = TemplateManager(PROJECT_ROOT / "templates")
    tm.save_metadata()
    created = tm.create_batch_stubs()
    logger.info("Template registry: %d stubs created", created)


def create_data_yaml() -> None:
    yaml_path = PROJECT_ROOT / "data.yaml"
    if yaml_path.exists():
        logger.info("SKIP data.yaml (exists)")
        return
    content = """path: ./dataset
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: eyewear
"""
    yaml_path.write_text(content, encoding="utf-8")
    logger.info("OK  data.yaml")


def create_gitignore() -> None:
    path = PROJECT_ROOT / ".gitignore"
    if path.exists():
        return
    path.write_text("""__pycache__/
*.pyc
.env
runs/
*.pt
*.onnx
dataset/images/
dataset/masks/
output/
.venv/
""", encoding="utf-8")
    logger.info("OK  .gitignore")


def ensure_geometric_template() -> None:
    glb = PROJECT_ROOT / "templates" / "geometric_metal.glb"
    if glb.exists():
        logger.info("OK  geometric_metal.glb exists")
        return
    logger.info("Generating geometric_metal.glb...")
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_template.py")], check=True)


def show_next_steps() -> None:
    print("""
======================================================================
 DEFIRMATION VTO TOOLKIT — READY
======================================================================

Next steps:

  1. Create stubs (if not done):
     python eyewear_vto_toolkit.py templates --create-stubs --list

  2. Add training images to dataset/images/ and masks to dataset/masks/

  3. Register samples + create YOLO splits:
     python eyewear_vto_toolkit.py dataset --create-splits

  4. Train segmentation:
     python train_yolov8_seg.py create-yaml
     python train_yolov8_seg.py train --dataset-yaml data.yaml --epochs 50

  5. Calibrate rim deformation (port 8001):
     python deformation_tuning_api.py
     Open http://localhost:8001

  6. Main VTO API (port 8000):
     uvicorn backend.api.main:app --reload --port 8000

See QUICK_START.md and TOOLKIT_GUIDE.md for full workflow.
""")


def main() -> None:
    print("\n=== Defirmation Toolkit Bootstrap ===\n")
    if not check_dependencies():
        sys.exit(1)
    create_directories()
    create_data_yaml()
    create_gitignore()
    ensure_geometric_template()
    init_templates()
    show_next_steps()


if __name__ == "__main__":
    main()
