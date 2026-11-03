#!/usr/bin/env python3
"""Read-only audit and visual contact sheets for the downloaded eyewear photo pool.

The script never edits, moves, renames or labels source images. It writes an
inspection report and a human annotation-review queue to a separate output
directory. The queue deliberately keeps every image UNLABELLED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MANIFESTS = ("train_manifest.csv", "validation_manifest.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
QUEUE_FIELDS = (
    "split", "local_image", "product_id", "source_label", "source_dataset",
    "source_license", "sha256", "annotation_status", "review_status",
    "view_type", "visible_classes", "reviewer", "review_notes",
)


def _read_manifests(dataset_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in MANIFESTS:
        path = dataset_dir / name
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["_manifest"] = name
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No {', '.join(MANIFESTS)} found in {dataset_dir}"
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path) -> dict:
    result = {"exists": path.is_file(), "decodable": False}
    if not path.is_file():
        result["error"] = "FILE_MISSING"
        return result
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        result["error"] = "UNSUPPORTED_EXTENSION"
        return result
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            result.update({
                "decodable": True,
                "width": width,
                "height": height,
                "mode": image.mode,
                "format": image.format,
                "aspect_ratio": round(width / height, 4) if height else None,
            })
    except Exception as exc:
        result["error"] = f"DECODE_ERROR: {type(exc).__name__}: {exc}"
    return result


def _make_contact_sheets(
    records: list[dict], output_dir: Path, thumbs_per_page: int
) -> list[str]:
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["inspection"].get("decodable"):
            groups[(record["split"], record["source_label"])].append(record)

    columns = 4
    thumb_w, thumb_h = 240, 190
    label_h, margin = 34, 12
    for (split, category), items in sorted(groups.items()):
        for page_index in range(0, len(items), thumbs_per_page):
            page_items = items[page_index:page_index + thumbs_per_page]
            rows = (len(page_items) + columns - 1) // columns
            canvas = Image.new(
                "RGB",
                (columns * (thumb_w + margin) + margin,
                 rows * (thumb_h + label_h + margin) + margin),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for index, record in enumerate(page_items):
                x = margin + (index % columns) * (thumb_w + margin)
                y = margin + (index // columns) * (thumb_h + label_h + margin)
                with Image.open(record["absolute_path"]) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((thumb_w, thumb_h))
                    px = x + (thumb_w - thumb.width) // 2
                    py = y + (thumb_h - thumb.height) // 2
                    canvas.paste(thumb, (px, py))
                caption = (
                    f'{record["product_id"] or "no-id"} | '
                    f'{record["local_image"]}'
                )
                draw.text((x, y + thumb_h + 3), caption[:52], fill="black")
            filename = (
                f"{split}_{category}_{page_index // thumbs_per_page + 1:03d}.jpg"
            )
            target = sheet_dir / filename
            canvas.save(target, quality=88)
            created.append(target.relative_to(output_dir).as_posix())
    return created


def inspect_dataset(
    dataset_dir: Path, output_dir: Path, thumbs_per_page: int = 24
) -> dict:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if dataset_dir == output_dir or dataset_dir in output_dir.parents:
        # Output may be a sibling folder inside the dataset root, but never
        # place it under images/ where a future run could ingest its own output.
        if output_dir == dataset_dir / "images" or dataset_dir / "images" in output_dir.parents:
            raise ValueError("Inspection output must not be inside the images directory.")
    rows = _read_manifests(dataset_dir)
    records: list[dict] = []
    seen_hashes: dict[str, list[dict]] = defaultdict(list)
    product_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts = Counter()
    for row in rows:
        relative = row.get("local_image", "")
        path = (dataset_dir / relative).resolve()
        try:
            path.relative_to(dataset_dir)
        except ValueError:
            inspection = {"exists": False, "decodable": False, "error": "PATH_TRAVERSAL"}
        else:
            inspection = _inspect_image(path)
        split = row.get("split", "")
        category = row.get("source_label", "")
        counts[(split, category)] += 1
        digest = row.get("sha256", "").strip().lower()
        actual_digest = _sha256(path) if inspection.get("decodable") else ""
        hash_matches = bool(digest and actual_digest and digest == actual_digest)
        record = {
            "split": split,
            "source_label": category,
            "local_image": relative,
            "product_id": row.get("product_id", ""),
            "expected_sha256": digest,
            "actual_sha256": actual_digest,
            "sha256_matches": hash_matches,
            "inspection": inspection,
            "absolute_path": path,
            "manifest": row.get("_manifest", ""),
        }
        records.append(record)
        if actual_digest:
            seen_hashes[actual_digest].append(record)
        if record["product_id"]:
            product_splits[(category, record["product_id"])].add(split)

    cross_split_duplicates = [
        {"sha256": digest, "images": [r["local_image"] for r in group]}
        for digest, group in seen_hashes.items()
        if len({r["split"] for r in group}) > 1
    ]
    product_leakage = [
        {"category": category, "product_id": product_id, "splits": sorted(splits)}
        for (category, product_id), splits in product_splits.items()
        if len(splits) > 1
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths = _make_contact_sheets(records, output_dir, thumbs_per_page)

    queue_path = output_dir / "annotation_review_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        for record in records:
            source_row = next(
                row for row in rows
                if row.get("local_image") == record["local_image"]
                and row.get("split") == record["split"]
            )
            writer.writerow({
                "split": record["split"],
                "local_image": record["local_image"],
                "product_id": record["product_id"],
                "source_label": record["source_label"],
                "source_dataset": source_row.get("source_dataset", ""),
                "source_license": source_row.get("source_license", ""),
                "sha256": record["actual_sha256"],
                "annotation_status": "UNLABELLED",
                "review_status": "NOT_REVIEWED",
                "view_type": "",
                "visible_classes": "",
                "reviewer": "",
                "review_notes": "",
            })

    report_records = []
    for record in records:
        report_records.append({
            key: value for key, value in record.items()
            if key not in {"absolute_path"}
        })
    report = {
        "dataset_dir": str(dataset_dir),
        "images_in_manifests": len(records),
        "images_existing": sum(r["inspection"].get("exists", False) for r in records),
        "images_decodable": sum(r["inspection"].get("decodable", False) for r in records),
        "sha256_matches": sum(r["sha256_matches"] for r in records),
        "sha256_mismatches_or_missing": sum(not r["sha256_matches"] for r in records),
        "counts_by_split_and_category": {
            f"{split}/{category}": count
            for (split, category), count in sorted(counts.items())
        },
        "unique_product_groups": len(product_splits),
        "cross_split_duplicate_images": cross_split_duplicates,
        "cross_split_product_leakage": product_leakage,
        "contact_sheets": sheet_paths,
        "annotation_review_queue": str(queue_path),
        "annotation_warning": (
            "This report does not create or approve segmentation masks. "
            "Every queue row remains UNLABELLED / NOT_REVIEWED."
        ),
        "records": report_records,
    }
    (output_dir / "inspection_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset root containing train_manifest.csv and validation_manifest.csv.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Separate report directory; defaults to <dataset>/inspection.")
    parser.add_argument("--thumbs-per-page", type=int, default=24)
    args = parser.parse_args()
    if args.thumbs_per_page < 1:
        parser.error("--thumbs-per-page must be at least 1")
    output = args.output or args.dataset / "inspection"
    try:
        report = inspect_dataset(args.dataset, output, args.thumbs_per_page)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({
        key: report[key] for key in (
            "images_in_manifests", "images_existing", "images_decodable",
            "sha256_matches", "sha256_mismatches_or_missing",
            "counts_by_split_and_category", "unique_product_groups",
            "cross_split_duplicate_images", "cross_split_product_leakage",
            "annotation_review_queue",
        )
    }, indent=2))
    print(f"Full report: {output.resolve() / 'inspection_report.json'}")
    print(f"Contact sheets: {output.resolve() / 'contact_sheets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
