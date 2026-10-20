#!/usr/bin/env python3
"""Production validation for the 1-class eyewear segmentation model.

Compares YOLOv8-Seg predictions against ground-truth YOLO polygon labels and
produces dataset-level and per-image metrics plus worst-case overlays.

Usage:
    python scripts/validate_binary_segmentation.py
    python scripts/validate_binary_segmentation.py --dataset-root dataset
    python scripts/validate_binary_segmentation.py --model models/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS_ID = 0
EXPECTED_CLASS_NAME = "eyewear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("validation_output"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--worst-k", type=int, default=10)
    return parser.parse_args()


def find_model(explicit: Path | None) -> Path:
    if explicit:
        return explicit

    candidates = [
        Path("models/glasses_seg.pt"),
        Path("models/best.pt"),
        Path("runs/segment/train/weights/best.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "No fine-tuned eyewear model found. Expected one of: "
        + ", ".join(map(str, candidates))
    )


def yolo_polygon_to_mask(
    label_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.is_file():
        return mask

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or len(values[1:]) % 2 != 0:
            raise ValueError(f"Invalid segmentation label {label_path}:{line_no}")

        class_id = int(values[0])
        if class_id != EXPECTED_CLASS_ID:
            raise ValueError(
                f"Unexpected class {class_id} in {label_path}:{line_no}; "
                f"expected {EXPECTED_CLASS_ID} ({EXPECTED_CLASS_NAME})"
            )

        coords = np.asarray([float(v) for v in values[1:]], dtype=np.float32).reshape(-1, 2)
        points = np.round(coords * np.array([width, height], dtype=np.float32)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)

    return mask


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0

    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    union = tp + fp + fn

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / union if union else 1.0

    pred_area = int(np.count_nonzero(p))
    truth_area = int(np.count_nonzero(t))
    area_error = abs(pred_area - truth_area) / truth_area if truth_area else float("inf")

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "area_error": area_error,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "pred_area_pixels": pred_area,
        "truth_area_pixels": truth_area,
    }


def boundary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred_u8 = (pred > 0).astype(np.uint8)
    truth_u8 = (truth > 0).astype(np.uint8)

    pred_boundary = cv2.morphologyEx(
        pred_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    truth_boundary = cv2.morphologyEx(
        truth_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )

    truth_count = int(truth_boundary.sum())
    pred_count = int(pred_boundary.sum())

    if truth_count == 0:
        return {"boundary_recall": 1.0 if pred_count == 0 else 0.0}

    distance = cv2.distanceTransform(
        (1 - pred_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distances = distance[truth_boundary > 0]

    boundary_recall = float(np.mean(distances <= 2.0))
    mean_boundary_distance = float(np.mean(distances)) if len(distances) else 0.0

    return {
        "boundary_recall_2px": boundary_recall,
        "mean_boundary_distance_px": mean_boundary_distance,
    }


def overlay(image: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = image.copy()
    p = pred > 0
    t = truth > 0

    # Red = false positive, blue = false negative, green = correct overlap.
    out[p & ~t] = (0, 0, 255)
    out[~p & t] = (255, 0, 0)
    out[p & t] = (
        0.55 * out[p & t] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    return out


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required. Install the project requirements first."
        ) from exc

    dataset_root = args.dataset_root.resolve()
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    output_dir = args.output.resolve()
    worst_dir = output_dir / "worst_cases"
    worst_dir.mkdir(parents=True, exist_ok=True)

    model_path = find_model(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))
    names = getattr(getattr(model, "model", None), "names", None)
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        names = {}

    if names != {0: EXPECTED_CLASS_NAME}:
        raise RuntimeError(
            f"MODEL CONTRACT FAILURE: {model_path} has classes {names!r}; "
            f"expected {{0: '{EXPECTED_CLASS_NAME}'}}."
        )

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No validation images found in {image_dir}")

    rows: list[dict[str, Any]] = []
    failures = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            failures += 1
            continue

        height, width = image.shape[:2]
        truth = yolo_polygon_to_mask(label_path, width, height)

        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        pred = np.zeros((height, width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                continue
            for mask_data in result.masks.data:
                mask = mask_data.detach().cpu().numpy().astype(np.float32)
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                pred = np.maximum(pred, (mask > 0.5).astype(np.uint8) * 255)

        metrics = binary_metrics(pred, truth)
        metrics.update(boundary_metrics(pred, truth))
        metrics["image"] = image_path.name
        metrics["label_exists"] = label_path.is_file()
        rows.append(metrics)

    if not rows:
        raise RuntimeError("No images could be evaluated.")

    metric_keys = [
        "iou", "dice", "precision", "recall", "area_error",
        "boundary_recall_2px", "mean_boundary_distance_px",
    ]

    aggregate = {
        key: float(np.mean([float(r[key]) for r in rows if np.isfinite(float(r[key]))]))
        for key in metric_keys
    }

    worst = sorted(rows, key=lambda r: (r["iou"], r["dice"]))[: args.worst_k]
    for index, row in enumerate(worst, 1):
        image_path = image_dir / row["image"]
        image = cv2.imread(str(image_path))
        truth = yolo_polygon_to_mask(
            label_dir / f"{image_path.stem}.txt",
            image.shape[1],
            image.shape[0],
        )
        results = model.predict(source=image, conf=args.conf, iou=args.iou, verbose=False)
        pred = np.zeros(truth.shape, dtype=np.uint8)
        for result in results:
            if result.masks is not None:
                for mask_data in result.masks.data:
                    mask = cv2.resize(
                        mask_data.detach().cpu().numpy().astype(np.float32),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    pred = np.maximum(pred, (mask > 0.5).astype(np.uint8) * 255)
        cv2.imwrite(
            str(worst_dir / f"{index:02d}_{Path(row['image']).stem}_overlay.jpg"),
            overlay(image, pred, truth),
        )

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(model_path),
        "dataset": str(dataset_root),
        "split": args.split,
        "images_evaluated": len(rows),
        "image_read_failures": failures,
        "aggregate_mean": aggregate,
        "worst_cases": [
            {
                "image": r["image"],
                "iou": r["iou"],
                "dice": r["dice"],
                "precision": r["precision"],
                "recall": r["recall"],
                "area_error": r["area_error"],
                "boundary_recall_2px": r.get("boundary_recall_2px"),
                "mean_boundary_distance_px": r.get("mean_boundary_distance_px"),
            }
            for r in worst
        ],
        "note": (
            "Metrics are measured against the repository's validation labels. "
            "They do not establish real-world production accuracy until the "
            "validation set contains representative real eyewear photography."
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                "BINARY EYEWEAR SEGMENTATION VALIDATION",
                f"Model: {model_path}",
                f"Images evaluated: {len(rows)}",
                "",
                *[
                    f"{key}: {value:.6f}"
                    for key, value in aggregate.items()
                ],
                "",
                "Worst cases are saved under worst_cases/.",
                summary["note"],
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
