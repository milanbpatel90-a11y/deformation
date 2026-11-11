#!/usr/bin/env python3
"""Validate image/mask/YOLO segmentation pairs for this repository.

Checks manifest file references, label syntax/class ID, normalized coordinates,
and optionally image/mask dimensions. It cannot certify visual accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def parse_label(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError("missing label file")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(f"line {number}: expected class plus at least 3 (x,y) points")
        if tokens[0] != "0":
            raise ValueError(f"line {number}: expected class 0, found {tokens[0]}")
        try:
            coords = [float(token) for token in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"line {number}: non-numeric polygon coordinate") from exc
        if any(not 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"line {number}: coordinates must be normalized to [0,1]")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Folder containing images/, masks/, labels/, and manifest.csv")
    p.add_argument("--check-mask-dimensions", action="store_true",
                   help="Compare image/mask dimensions (requires opencv-python)")
    p.add_argument("--json", type=Path, help="Optional JSON report path")
    args = p.parse_args()
    root = args.dataset_root.resolve()
    report = {"dataset_root": str(root), "images": 0, "valid_labels": 0, "valid_masks": 0,
              "errors": [], "warnings": [], "structural_only": True}
    manifest = root / "manifest.csv"
    rows = []
    if not manifest.is_file():
        report["errors"].append("manifest.csv not found")
    else:
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        report["manifest_rows"] = len(rows)
    split_counts = {}
    for row in rows:
        image_id = row.get("image_id") or "<unknown>"
        try:
            paths = [(root / row[key]).resolve() for key in ("local_image", "mask_path", "label_path")]
        except KeyError as exc:
            report["errors"].append(f"{image_id}: missing manifest column {exc}")
            continue
        if any(not path.is_relative_to(root) for path in paths):
            report["errors"].append(f"{image_id}: a manifest path escapes the dataset root")
            continue
        image_path, mask_path, label_path = paths
        split = row.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
        if not image_path.is_file():
            report["errors"].append(f"{image_id}: image missing ({image_path.relative_to(root)})")
            continue
        report["images"] += 1
        if mask_path.is_file():
            report["valid_masks"] += 1
        else:
            report["errors"].append(f"{image_id}: mask missing ({mask_path.relative_to(root)})")
        try:
            parse_label(label_path)
            report["valid_labels"] += 1
        except Exception as exc:
            report["errors"].append(f"{image_id}: {exc}")
        if args.check_mask_dimensions:
            if cv2 is None:
                report["errors"].append("--check-mask-dimensions requires opencv-python")
                break
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
            if image is None or mask is None:
                report["errors"].append(f"{image_id}: unable to read image or mask")
            elif image.shape[:2] != mask.shape[:2]:
                report["errors"].append(f"{image_id}: image/mask dimensions differ {image.shape[:2]} vs {mask.shape[:2]}")
    report["split_counts"] = split_counts
    if report.get("manifest_rows", 0) != 1000:
        report["warnings"].append(f"Expected 1,000 manifest rows; found {report.get('manifest_rows', 0)}")
    if report.get("manifest_rows", 0) and report["valid_labels"] != report["manifest_rows"]:
        report["warnings"].append("Not all rows have valid YOLO polygon labels")
    report["result"] = "PASS_STRUCTURE_ONLY" if not report["errors"] else "FAIL"
    report["note"] = "Structural validation does not prove visual mask accuracy. Human review is required; keep test data out of training."
    output = json.dumps(report, indent=2)
    print(output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
