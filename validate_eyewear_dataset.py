#!/usr/bin/env python3
"""Validate image/mask/YOLO segmentation pairs for the deformation repository.

Checks file references, binary mask presence, normalized YOLO polygon syntax and
optionally image/mask dimensions. It cannot certify visual segmentation accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def parse_label(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError("missing label file")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_no, line in enumerate(lines, 1):
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"line {line_no}: expected class plus at least 3 polygon (x,y) points")
        if values[0] != "0":
            raise ValueError(f"line {line_no}: expected class 0, found {values[0]}")
        try:
            coords = [float(value) for value in values[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_no}: non-numeric polygon coordinate") from exc
        if any(not 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"line {line_no}: coordinates must be normalized to [0,1]")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Folder containing images/, masks/, labels/ and manifest.csv")
    parser.add_argument("--check-mask-dimensions", action="store_true",
                        help="Compare image/mask dimensions (requires opencv-python)")
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    report = {
        "dataset_root": str(root), "images": 0, "valid_labels": 0,
        "valid_masks": 0, "errors": [], "warnings": [], "structural_only": True,
    }
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        report["errors"].append("manifest.csv not found")
        rows = []
    else:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report["manifest_rows"] = len(rows)

    split_counts: dict[str, int] = {}
    for row in rows:
        image_id = row.get("image_id") or "<unknown>"
        try:
            img = (root / row["local_image"]).resolve()
            mask = (root / row["mask_path"]).resolve()
            label = (root / row["label_path"]).resolve()
        except KeyError as exc:
            report["errors"].append(f"{image_id}: missing manifest column {exc}")
            continue
        for path in (img, mask, label):
            if not path.is_relative_to(root):
                report["errors"].append(f"{image_id}: manifest path escapes dataset root")
                break
        else:
            split = row.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1
            if not img.is_file():
                report["errors"].append(f"{image_id}: image missing ({img.relative_to(root)})")
                continue
            report["images"] += 1
            if not mask.is_file():
                report["errors"].append(f"{image_id}: mask missing ({mask.relative_to(root)})")
            else:
                report["valid_masks"] += 1
            try:
                parse_label(label)
                report["valid_labels"] += 1
            except Exception as exc:
                report["errors"].append(f"{image_id}: {exc}")
            if args.check_mask_dimensions:
                if cv2 is None:
                    report["errors"].append("--check-mask-dimensions requires opencv-python")
                    break
                image_data = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
                mask_data = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) if mask.is_file() else None
                if image_data is None or mask_data is None:
                    report["errors"].append(f"{image_id}: unable to read image or mask")
                elif image_data.shape[:2] != mask_data.shape[:2]:
                    report["errors"].append(
                        f"{image_id}: image/mask dimensions differ: "
                        f"{image_data.shape[:2]} vs {mask_data.shape[:2]}"
                    )
    report["split_counts"] = split_counts
    if report.get("manifest_rows", 0) != 1000:
        report["warnings"].append(
            f"Expected 1,000 rows for this dataset; found {report.get('manifest_rows', 0)}"
        )
    if report.get("manifest_rows", 0) and report["valid_labels"] != report["manifest_rows"]:
        report["warnings"].append("Not all rows have structurally valid labels")
    report["result"] = "PASS_STRUCTURE_ONLY" if not report["errors"] else "FAIL"
    report["note"] = (
        "Structural validation does not prove visual mask quality. AI prelabels must be "
        "human-reviewed; keep the test split out of training."
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
