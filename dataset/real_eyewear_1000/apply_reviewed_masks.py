#!/usr/bin/env python3
"""Apply reviewed masks and synchronize YOLO polygons + manifest status.

Audit format matches HUMAN_REVIEW_TOOL_FIXED.html:
  {"decisions":[{"image_id":"real_000001","split":"train",
    "status":"approved|corrected|rejected","reviewer":"...", ...}]}

For corrected masks, supply files named <image_id>_mask.png in --review-masks.
Runs in dry-run mode by default; --apply is required to modify the dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def polygon_lines(mask, min_area: float = 3.0) -> list[str]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install requirements-eyewear-dataset.txt")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    lines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        contour = cv2.approxPolyDP(contour, max(0.8, 0.0015 * cv2.arcLength(contour, True)), True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        coordinates = []
        for x, y in contour:
            coordinates.extend((min(1.0, max(0.0, float(x) / max(1, width - 1))),
                                min(1.0, max(0.0, float(y) / max(1, height - 1))))
        lines.append("0 " + " ".join(f"{value:.6f}" for value in coordinates))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("."))
    parser.add_argument("--review-masks", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Actually change dataset files; without this only validates.")
    args = parser.parse_args()
    root = args.dataset.resolve()
    corrections = args.review_masks.resolve()
    audit_path = args.audit_json.resolve()
    if not (root / "manifest.csv").is_file():
        parser.error(f"manifest.csv not found under {root}")
    if not audit_path.is_file():
        parser.error(f"audit JSON not found: {audit_path}")
    if not corrections.is_dir():
        parser.error(f"corrected-mask directory not found: {corrections}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        decisions = audit.get("decisions", [])
    except Exception as exc:
        parser.error(f"Unable to parse audit JSON: {exc}")

    keyed = {}
    errors = []
    for decision in decisions:
        split = decision.get("split")
        image_id = decision.get("image_id")
        status = decision.get("status")
        reviewer = (decision.get("reviewer") or "").strip()
        if not image_id or split not in {"train", "val", "test"}:
            errors.append(f"Invalid audit row: missing image_id or invalid split: {decision}")
        elif not reviewer:
            errors.append(f"{image_id}: reviewer is required")
        elif status not in {"approved", "corrected", "rejected"}:
            errors.append(f"{image_id}: unsupported review status {status}")
        else:
            key = (split, image_id)
            if key in keyed:
                errors.append(f"{image_id}: duplicate audit decision for split {split}")
            keyed[key] = decision

    manifest_file = root / "manifest.csv"
    with manifest_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    if not fieldnames:
        parser.error("manifest.csv has no header")
    indexed = {(row.get("split"), row.get("image_id")): row for row in rows}
    missing = [key for key in keyed if key not in indexed]
    errors.extend(f"{key[1]}: audit entry does not match manifest split {key[0]}" for key in missing)
    actionable = []
    for key, decision in keyed.items():
        if key not in indexed:
            continue
        status = decision["status"]
        row = indexed[key]
        if status == "rejected":
            actionable.append((key, row, decision, None, None, None))
        elif status == "corrected":
            image_id = key[1]
            source_mask = corrections / f"{image_id}_mask.png"
            if not source_mask.is_file():
                errors.append(f"{image_id}: corrected status but PNG missing at {source_mask}")
                continue
            if cv2 is None:
                errors.append("OpenCV is required to read masks and generate labels")
                break
            mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            image_path = root / row["local_image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if mask is None or image is None:
                errors.append(f"{image_id}: could not read original image or corrected mask")
                continue
            if mask.shape[:2] != image.shape[:2]:
                errors.append(f"{image_id}: corrected mask size {mask.shape[:2]} does not match image {image.shape[:2]}")
                continue
            lines = polygon_lines(mask)
            if not lines:
                errors.append(f"{image_id}: corrected mask contains no nontrivial foreground polygon")
                continue
            actionable.append((key, row, decision, source_mask, mask, lines))
        else:
            actionable.append((key, row, decision, None, None, None))

    print(f"Audit entries: {len(decisions)}; matched decisions: {len(actionable)}; errors: {len(errors)}")
    for error in errors[:50]:
        print("ERROR:", error)
    if errors:
        return 2
    if not args.apply:
        print("DRY RUN ONLY. No files modified. Re-run with --apply after reviewing the planned changes.")
        return 0

    backup_root = root / "human_review_backups"
    corrected_count = 0
    applied_count = 0
    rejected_count = 0
    for key, row, decision, source_mask, mask, lines in actionable:
        split, image_id = key
        status = decision["status"]
        if status == "rejected":
            row["annotation_status"] = "HUMAN_REVIEW_REJECTED"
            rejected_count += 1
        elif status == "corrected":
            mask_dest = root / row["mask_path"]
            label_dest = root / row["label_path"]
            mask_backup = backup_root / row["mask_path"]
            label_backup = backup_root / row["label_path"]
            mask_backup.parent.mkdir(parents=True, exist_ok=True)
            label_backup.parent.mkdir(parents=True, exist_ok=True)
            if mask_dest.exists():
                shutil.copy2(mask_dest, mask_backup)
            if label_dest.exists():
                shutil.copy2(label_dest, label_backup)
            shutil.copy2(source_mask, mask_dest)
            label_dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            row["annotation_status"] = "HUMAN_VERIFIED_CORRECTED"
            corrected_count += 1
            applied_count += 1
        else:
            row["annotation_status"] = "HUMAN_VERIFIED_ACCEPTED_PRELABEL"
            applied_count += 1
        row["reviewer"] = decision["reviewer"]
        row["reviewed_at"] = decision.get("time", "")

    fields = list(fieldnames)
    for name in ("reviewer", "reviewed_at"):
        if name not in fields:
            fields.extend([name])
    temp = manifest_file.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(manifest_file)
    print(f"Applied {applied_count} reviewed decisions; {corrected_count} corrected masks and labels updated; {rejected_count} marked rejected.")
    print(f"Pre-change backups for corrected masks/labels: {backup_root}")
    print("Re-run validate_eyewear_dataset.py and visually inspect results before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
