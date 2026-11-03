#!/usr/bin/env python3
"""Run a validated eyewear YOLO checkpoint on real product photos and save QA overlays.

This is an inference/visual-review tool, not an accuracy benchmark. It records
class coverage and mask area for each input but does not create ground-truth
labels or claim precision/recall.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from backend.segmentation.segmenter import GlassesSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_PARTS = {
    "eyewear_rim",
    "eyewear_temple",
    "bridge",
    "left_lens",
    "right_lens",
    "nose_pad",
}
COLORS = {
    "eyewear": (0, 220, 0),
    "eyewear_rim": (0, 220, 0),
    "eyewear_temple": (255, 120, 0),
    "bridge": (0, 180, 255),
    "left_lens": (255, 0, 180),
    "right_lens": (180, 0, 255),
    "nose_pad": (0, 255, 255),
}


def _images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path does not exist: {input_path}")
    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def preview(input_path: Path, output_dir: Path, model_path: Path, confidence: float) -> dict:
    segmenter = GlassesSegmenter(
        model_path=model_path,
        require_model=True,
        confidence_threshold=confidence,
    )
    input_images = _images(input_path)
    if not input_images:
        raise ValueError(f"No supported product images found under {input_path}")

    records = []
    overlay_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"

    for image_path in input_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            records.append({
                "image": str(image_path),
                "status": "ERROR",
                "error": "OpenCV could not decode image",
            })
            continue

        result = segmenter.segment(image)
        part_masks = result.get("part_masks", {})
        overlay = image.copy()
        class_metrics = {}
        missing_parts = []

        for class_name, mask in part_masks.items():
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"{image_path}: mask {class_name} has shape {mask.shape}; "
                    f"expected {image.shape[:2]}"
                )
            area_px = int(np.count_nonzero(mask))
            area_pct = 100.0 * area_px / max(mask.size, 1)
            class_metrics[class_name] = {
                "area_pixels": area_px,
                "image_area_percent": round(area_pct, 4),
            }
            if area_px:
                color = COLORS.get(class_name, (255, 255, 255))
                colored = np.zeros_like(image)
                colored[mask > 0] = color
                overlay = cv2.addWeighted(overlay, 0.72, colored, 0.28, 0)

            mask_name = f"{image_path.stem}_{class_name}.png"
            _write_image(masks_dir / mask_name, mask.astype(np.uint8))

        if result.get("supports_part_masks"):
            missing_parts = sorted(
                name for name in EXPECTED_PARTS
                if class_metrics.get(name, {}).get("area_pixels", 0) == 0
            )
            status = "REVIEW_REQUIRED" if missing_parts else "PART_MASKS_PRESENT_REVIEW_VISUALLY"
        elif result.get("class_count") == 1:
            status = "SILHOUETTE_ONLY"
        else:
            status = "INVALID_NO_YOLO_PART_MODEL"

        combined = np.asarray(result.get("front"))
        combined_area = int(np.count_nonzero(combined)) if combined is not None else 0
        combined_pct = 100.0 * combined_area / max(image.shape[0] * image.shape[1], 1)

        _write_image(overlay_dir / f"{image_path.stem}_overlay.jpg", overlay)
        records.append({
            "image": str(image_path),
            "status": status,
            "model_type": result.get("model_type"),
            "class_count": result.get("class_count"),
            "supports_part_masks": bool(result.get("supports_part_masks")),
            "detections": result.get("detections"),
            "combined_mask_area_percent": round(combined_pct, 4),
            "missing_parts": missing_parts,
            "classes": class_metrics,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path": str(model_path.resolve()),
        "model_type": segmenter.model_type,
        "class_count": segmenter.model_class_count,
        "supports_part_masks": segmenter.supports_part_masks,
        "confidence_threshold": confidence,
        "images_total": len(input_images),
        "images_decoded": sum(1 for item in records if "error" not in item),
        "records": records,
        "warning": (
            "Mask coverage and overlays are review aids only. They are not accuracy metrics. "
            "Compare against human-reviewed ground truth before production use."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image", "status", "model_type", "class_count", "detections",
                "combined_mask_area_percent", "missing_parts",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(record.get(key)) if isinstance(record.get(key), list) else record.get(key)
                for key in writer.fieldnames
            })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One product image or a directory (recursive).")
    parser.add_argument("--model", type=Path, required=True, help="Validated fine-tuned eyewear YOLO checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("output/eyewear_segmentation_review"))
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    try:
        report = preview(args.input, args.output, args.model, args.confidence)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    print(json.dumps({
        "model_type": report["model_type"],
        "class_count": report["class_count"],
        "supports_part_masks": report["supports_part_masks"],
        "images_total": report["images_total"],
        "images_decoded": report["images_decoded"],
        "output": str(args.output.resolve()),
        "note": report["warning"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
