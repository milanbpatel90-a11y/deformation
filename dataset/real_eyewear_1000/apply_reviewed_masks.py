#!/usr/bin/env python3
"""Apply reviewed masks and synchronize YOLO polygons + manifest status.

Audit JSON contains decisions with image_id, split, status, reviewer and time.
Corrected masks must be named <image_id>_mask.png inside --review-masks.
No changes are made unless --apply is passed.
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


def polygon_lines(mask):
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install requirements-eyewear-dataset.txt")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    output = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 3:
            continue
        contour = cv2.approxPolyDP(
            contour, max(0.8, 0.0015 * cv2.arcLength(contour, True)), True
        ).reshape(-1, 2)
        if len(contour) < 3:
            continue
        coordinates = []
        for x, y in contour:
            coordinates.extend([
                min(1.0, max(0.0, float(x) / max(1, width - 1))),
                min(1.0, max(0.0, float(y) / max(1, height - 1))),
            ])
        output.append("0 " + " ".join(f"{value:.6f}" for value in coordinates))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("."))
    parser.add_argument("--review-masks", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Modify dataset files; default is dry-run")
    args = parser.parse_args()
    root = args.dataset.resolve()
    corrected_dir = args.review_masks.resolve()
    audit_path = args.audit_json.resolve()
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        parser.error(f"manifest.csv not found under {root}")
    if not audit_path.is_file():
        parser.error(f"audit JSON not found: {audit_path}")
    if not corrected_dir.is_dir():
        parser.error(f"corrected-mask directory not found: {corrected_dir}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        decisions = audit.get("decisions", [])
    except Exception as exc:
        parser.error(f"cannot parse audit JSON: {exc}")
    if not isinstance(decisions, list):
        parser.error("audit decisions must be a JSON list")

    keyed, errors = {}, []
    for decision in decisions:
        split = decision.get("split")
        image_id = decision.get("image_id")
        status = decision.get("status")
        reviewer = (decision.get("reviewer") or "").strip()
        if not image_id or split not in {"train", "val", "test"}:
            errors.append(f"invalid audit row: {decision}")
            continue
        if not reviewer:
            errors.append(f"{image_id}: reviewer is required")
            continue
        if status not in {"approved", "corrected", "rejected"}:
            errors.append(f"{image_id}: invalid status {status}")
            continue
        key = (split, image_id)
        if key in keyed:
            errors.append(f"{image_id}: duplicate decisions for split {split}")
            continue
        keyed[key] = decision

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not fields:
        parser.error("manifest.csv has no header")
    indexed = {(row.get("split"), row.get("image_id")): row for row in rows}
    for key in keyed:
        if key not in indexed:
            errors.append(f"{key[1]}: no manifest row for split {key[0]}")

    plan = []
    for key, decision in keyed.items():
        if key not in indexed:
            continue
        row = indexed[key]
        status = decision["status"]
        if status == "corrected":
            source_mask = corrected_dir / f"{key[1]}_mask.png"
            if not source_mask.is_file():
                errors.append(f"{key[1]}: corrected PNG missing: {source_mask}")
                continue
            if cv2 is None:
                errors.append("OpenCV is required to apply corrected masks")
                break
            corrected = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(root / row["local_image"]), cv2.IMREAD_UNCHANGED)
            if corrected is None or image is None:
                errors.append(f"{key[1]}: source image or mask cannot be read")
                continue
            if corrected.shape[:2] != image.shape[:2]:
                errors.append(f"{key[1]}: image/mask dimensions differ")
                continue
            try:
                labels = polygon_lines(corrected)
            except Exception as exc:
                errors.append(f"{key[1]}: {exc}")
                continue
            if not labels:
                errors.append(f"{key[1]}: corrected mask has no nontrivial polygons")
                continue
            plan.append((key, row, decision, source_mask, labels))
        else:
            plan.append((key, row, decision, None, None))

    print(f"Audit decisions={len(decisions)}; valid matches={len(plan)}; errors={len(errors)}")
    for error in errors[:50]:
        print("ERROR:", error)
    if errors:
        return 2
    if not args.apply:
        print("DRY RUN ONLY. No files modified. Re-run with --apply after reviewing.")
        return 0

    backup_root = root / "human_review_backups"
    applied = corrected_count = rejected_count = 0
    for key, row, decision, source_mask, labels in plan:
        status = decision["status"]
        if status == "rejected":
            row["annotation_status"] = "HUMAN_REVIEW_REJECTED"
            rejected_count += 1
        elif status == "corrected":
            mask_dest = root / row["mask_path"]
            label_dest = root / row["label_path"]
            for dest in (mask_dest, label_dest):
                if dest.exists():
                    backup = backup_root / dest.relative_to(root)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, backup)
            shutil.copy2(source_mask, mask_dest)
            label_dest.write_text("\n".join(labels) + "\n", encoding="utf-8")
            row["annotation_status"] = "HUMAN_VERIFIED_CORRECTED"
            corrected_count += 1
            applied += 1
        else:
            row["annotation_status"] = "HUMAN_VERIFIED_ACCEPTED_PRELABEL"
            applied += 1
        row["reviewer"] = decision["reviewer"]
        row["reviewed_at"] = decision.get("time", "")

    fields = list(fields)
    for name in ("reviewer", "reviewed_at"):
        if name not in fields:
            fields.append(name)
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest_path)
    print(f"Applied {applied}; corrected masks/labels {corrected_count}; rejected {rejected_count}.")
    print(f"Backups: {backup_root}")
    print("Re-run validate_eyewear_dataset.py and visually inspect before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
