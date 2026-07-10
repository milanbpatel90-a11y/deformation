"""
YOLOv8-Seg Training Script for Eyewear Segmentation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def validate_dataset_structure(dataset_root: str | Path) -> bool:
    dataset_path = Path(dataset_root)
    splits_dir = dataset_path / "splits"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not splits_dir.exists():
        logger.error("splits/ directory not found in %s", dataset_root)
        return False

    ok = True
    for split in ["train", "val"]:
        manifest = splits_dir / split / "manifest.json"
        if not manifest.exists():
            logger.error("Missing %s/manifest.json", split)
            ok = False
            continue
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        count = len(data.get("entries", []))
        if count < 10:
            logger.warning("  %s: only %d samples (recommend 50+)", split, count)
        else:
            logger.info("  %s: %d samples", split, count)

        img_count = len(list((images_dir / split).glob("*"))) if (images_dir / split).exists() else 0
        lbl_count = len(list((labels_dir / split).glob("*.txt"))) if (labels_dir / split).exists() else 0
        if img_count == 0:
            logger.warning("  %s: no images in images/%s/ — run dataset --create-splits first", split, split)
        else:
            logger.info("  %s: %d images, %d labels", split, img_count, lbl_count)

    if ok:
        logger.info("Dataset structure validated")
    return ok


def create_yolo_dataset_yaml(
    dataset_root: str | Path,
    output_path: str | Path = "data.yaml",
) -> Path:
    dataset_path = Path(dataset_root).resolve()
    yaml_content = f"""# YOLOv8 Segmentation — eyewear
path: {dataset_path.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: eyewear
"""
    output = Path(output_path)
    output.write_text(yaml_content, encoding="utf-8")
    logger.info("Created %s", output)
    return output


def train_yolov8_seg(
    dataset_yaml: str,
    model_name: str = "yolov8n-seg",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project_dir: str = "./runs/segment",
    name: str = "eyewear_seg",
    patience: int = 10,
    resume: bool = False,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return {}

    logger.info("Loading model: %s", model_name)
    model = YOLO(f"{model_name}.pt")

    logger.info("Starting training (epochs=%d, batch=%d)", epochs, batch_size)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project_dir,
        name=name,
        patience=patience,
        save=True,
        val=True,
        resume=resume,
        task="segment",
    )

    save_dir = getattr(results, "save_dir", Path(project_dir) / name)
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "results_dir": str(save_dir),
    }


def evaluate_model_on_dataset(
    model_path: str,
    dataset_yaml: str,
    img_size: int = 640,
    device: str = "0",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return {}

    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, imgsz=img_size, device=device)

    results = {
        "mAP50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
        "mAP50_95": float(metrics.box.map) if hasattr(metrics, "box") else None,
        "segmentation_mAP50": float(metrics.seg.map50) if hasattr(metrics, "seg") else None,
        "segmentation_mAP50_95": float(metrics.seg.map) if hasattr(metrics, "seg") else None,
    }
    for k, v in results.items():
        if v is not None:
            logger.info("  %s: %.4f", k, v)
    return results


def export_onnx(model_path: str, output_path: str = "model.onnx") -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return ""

    model = YOLO(model_path)
    exported = model.export(format="onnx", imgsz=640)
    logger.info("Exported to %s", exported)
    return str(exported)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8-Seg Training Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))

    yaml_parser = subparsers.add_parser("create-yaml")
    yaml_parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))
    yaml_parser.add_argument("--output", default="data.yaml")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-yaml", required=True)
    train_parser.add_argument("--model", default="yolov8n-seg")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--img-size", type=int, default=640)
    train_parser.add_argument("--device", default="0")
    train_parser.add_argument("--name", default="eyewear_seg")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--dataset-yaml", required=True)
    eval_parser.add_argument("--device", default="0")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model-path", required=True)
    export_parser.add_argument("--output", default="model.onnx")

    args = parser.parse_args()

    if args.command == "validate":
        validate_dataset_structure(args.dataset_root)

    elif args.command == "create-yaml":
        create_yolo_dataset_yaml(args.dataset_root, args.output)

    elif args.command == "train":
        yaml_path = Path(args.dataset_yaml)
        dataset_root = yaml_path.parent / "dataset"
        if not dataset_root.exists():
            dataset_root = PROJECT_ROOT / "dataset"
        validate_dataset_structure(dataset_root)
        metadata = train_yolov8_seg(
            args.dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            name=args.name,
        )
        print(json.dumps(metadata, indent=2))

    elif args.command == "evaluate":
        metrics = evaluate_model_on_dataset(args.model_path, args.dataset_yaml, device=args.device)
        print(json.dumps(metrics, indent=2))

    elif args.command == "export":
        export_onnx(args.model_path, args.output)


if __name__ == "__main__":
    main()
