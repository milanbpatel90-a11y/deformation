#!/usr/bin/env python3
"""Train a six-part YOLO segmentation model only from human-reviewed labels.

Required data.yaml class order:
0 eyewear_rim, 1 eyewear_temple, 2 bridge, 3 left_lens,
4 right_lens, 5 nose_pad.

The script validates YOLO polygons, review status and product-group split
separation before importing Ultralytics or starting training. It intentionally
does not convert one-class whole-object masks into fake part labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

EXPECTED_NAMES = [
    "eyewear_rim",
    "eyewear_temple",
    "bridge",
    "left_lens",
    "right_lens",
    "nose_pad",
]
HUMAN_VERIFIED = {
    "HUMAN_VERIFIED_ACCEPTED_PRELABEL",
    "HUMAN_VERIFIED_CORRECTED",
    "APPROVED",
    "CORRECTED",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _class_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        try:
            pairs = sorted((int(key), str(value)) for key, value in raw.items())
        except (TypeError, ValueError) as exc:
            raise ValueError("data.yaml names must use integer class IDs.") from exc
        if [key for key, _ in pairs] != list(range(len(pairs))):
            raise ValueError("data.yaml class IDs must be contiguous from zero.")
        return [value for _, value in pairs]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    raise ValueError("data.yaml must define names as an ID map or ordered list.")


def parse_polygon_line(line: str, line_number: int, num_classes: int = 6) -> int:
    tokens = line.split()
    if len(tokens) < 7 or (len(tokens) - 1) % 2:
        raise ValueError(f"line {line_number}: expected class ID and at least 3 polygon points")
    try:
        class_id = int(tokens[0])
        coords = [float(value) for value in tokens[1:]]
    except ValueError as exc:
        raise ValueError(f"line {line_number}: non-numeric class ID or coordinate") from exc
    if not 0 <= class_id < num_classes:
        raise ValueError(f"line {line_number}: class ID {class_id} outside [0,{num_classes - 1}]")
    if any(not 0.0 <= value <= 1.0 for value in coords):
        raise ValueError(f"line {line_number}: polygon coordinates must be normalized to [0,1]")

    points = list(zip(coords[0::2], coords[1::2]))
    if len(set(points)) < 3:
        raise ValueError(f"line {line_number}: polygon must contain at least 3 distinct points")
    twice_area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        twice_area += x1 * y2 - x2 * y1
    if abs(twice_area) <= 1e-8:
        raise ValueError(f"line {line_number}: polygon has zero or negligible area")
    return class_id


def _split_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"Image split directory does not exist: {path}")
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_review_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Human-review manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Human-review manifest is empty.")

    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        source = (
            row.get("local_image")
            or row.get("image_path")
            or row.get("image")
            or row.get("image_id")
            or ""
        )
        if not split or not source:
            raise ValueError("Review manifest needs split and local_image/image_path/image/image_id columns.")
        key = (split, Path(str(source)).stem)
        if key in indexed:
            raise ValueError(f"Duplicate image in review manifest: {key}")
        indexed[key] = row
    return indexed


def validate_dataset(data_yaml: Path, review_manifest: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML before validating the YOLO dataset.") from exc

    if not data_yaml.is_file():
        raise ValueError(f"YOLO data YAML does not exist: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YOLO data YAML must contain a mapping.")
    names = _class_names(config.get("names"))
    if names != EXPECTED_NAMES:
        raise ValueError(
            "Six-part class contract mismatch. Expected exactly "
            + repr(EXPECTED_NAMES)
            + "; received "
            + repr(names)
            + ". Do not train with a one-class silhouette YAML."
        )

    root_value = config.get("path")
    dataset_root = Path(root_value) if root_value else data_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    review_rows = _load_review_rows(review_manifest)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "class_names": names,
        "review_manifest": str(review_manifest.resolve()),
        "splits": {},
        "group_split_leakage": [],
    }
    group_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        configured_images = config.get(split)
        if not configured_images:
            if split == "test":
                continue
            raise ValueError(f"data.yaml is missing the required {split!r} split.")
        if isinstance(configured_images, list):
            raise ValueError(f"Split {split} must point to one directory, not a list.")
        image_root = Path(str(configured_images))
        if not image_root.is_absolute():
            image_root = (dataset_root / image_root).resolve()
        images = _split_images(image_root)
        if not images:
            raise ValueError(f"No supported images found for {split}: {image_root}")

        label_root = dataset_root / "labels" / split
        class_counts = {str(class_id): 0 for class_id in range(6)}
        reviewed_count = 0
        for image_path in images:
            relative = image_path.relative_to(image_root)
            label_path = label_root / relative.with_suffix(".txt")
            if not label_path.is_file():
                raise ValueError(f"Missing YOLO label for {split}/{relative}: {label_path}")
            lines = [
                line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                raise ValueError(f"Empty segmentation label file: {label_path}")
            for line_number, line in enumerate(lines, 1):
                class_id = parse_polygon_line(line, line_number)
                class_counts[str(class_id)] += 1

            review = review_rows.get((split, image_path.stem))
            if review is None:
                raise ValueError(f"Image is missing from human-review manifest: {split}/{image_path.name}")
            status = str(
                review.get("annotation_status")
                or review.get("review_status")
                or review.get("status")
                or ""
            ).strip().upper()
            if status not in HUMAN_VERIFIED:
                raise ValueError(
                    f"Refusing to train on unreviewed image {image_path.name}: "
                    f"annotation status is {status or 'missing'}."
                )
            reviewed_count += 1

            group = str(
                review.get("product_id")
                or review.get("product_model_id")
                or review.get("product_group")
                or ""
            ).strip()
            if group:
                previous = group_to_split.get(group)
                if previous and previous != split:
                    report["group_split_leakage"].append(
                        {"product_group": group, "splits": [previous, split]}
                    )
                else:
                    group_to_split[group] = split

        report["splits"][split] = {
            "images": len(images),
            "human_reviewed_images": reviewed_count,
            "polygon_counts_by_class": class_counts,
        }
        if split in {"train", "val"}:
            missing_classes = [key for key, count in class_counts.items() if count == 0]
            if missing_classes:
                raise ValueError(
                    f"{split} split has no polygon labels for class IDs {missing_classes}; "
                    "six-class training/evaluation would be incomplete."
                )

    if report["group_split_leakage"]:
        raise ValueError(
            "Product groups appear in multiple splits; fix the split before training: "
            + json.dumps(report["group_split_leakage"])
        )
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    report = validate_dataset(args.data, args.review_manifest)
    if args.validate_only:
        return {"validation": report, "training": "not_started"}

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before training: pip install ultralytics") from exc

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data.resolve()),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=42,
        deterministic=True,
        cache=False,
        pretrained=True,
        close_mosaic=10,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=True,
        val=True,
    )
    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Training finished but best checkpoint was not produced: {best_path}")

    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(data=str(args.data.resolve()), split="val", imgsz=args.imgsz, device=args.device)
    metrics: dict[str, Any] = {}
    seg = getattr(val_metrics, "seg", None)
    for key, attribute in (
        ("mask_map50", "map50"),
        ("mask_map50_95", "map"),
        ("mask_precision", "p"),
        ("mask_recall", "r"),
    ):
        value = getattr(seg, attribute, None) if seg is not None else None
        if value is not None:
            # Ultralytics returns scalar mAP values and per-class precision /
            # recall arrays. Preserve arrays as JSON numbers rather than
            # turning them into lossy string representations.
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                metrics[key] = [float(item) for item in value]
            else:
                metrics[key] = float(value)
    metrics["class_order"] = list(EXPECTED_NAMES)

    report_path = save_dir / "real_world_segmentation_report.json"
    report_path.write_text(json.dumps({
        "dataset_validation": report,
        "checkpoint": str(best_path.resolve()),
        "validation_metrics": metrics,
        "test_split_used": False,
        "note": "Validation metrics are meaningful only if every mask is visually human-reviewed.",
    }, indent=2) + "\n", encoding="utf-8")
    return {
        "validation": report,
        "checkpoint": str(best_path.resolve()),
        "validation_metrics": metrics,
        "report": str(report_path.resolve()),
        "test_split_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Six-class YOLO data.yaml")
    parser.add_argument("--review-manifest", type=Path, required=True,
                        help="CSV containing split, image path/ID and human review status")
    parser.add_argument("--weights", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="Use 'cpu' or a CUDA device such as '0'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    parser.add_argument("--name", default="eyewear_parts_real")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        result = train(args)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
