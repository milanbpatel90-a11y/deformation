#!/usr/bin/env python3
"""Stream real product eyewear photos from Hugging Face for manual part annotation.

This downloads images only. It does NOT fabricate segmentation labels. The
source is CC-BY-4.0 and product-model-aware train/validation splits are kept
separate. Use the images for CVAT/X-AnyLabeling annotation before YOLO training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
from typing import Any


DATASET_ID = "shravya11/eyeglasses-dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"


def _decode_image(value: Any):
    from PIL import Image

    if hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
    raise ValueError("Dataset row does not contain a decodable image.")


def _label_name(dataset, label_id: Any) -> str:
    feature = dataset.features.get("label") if getattr(dataset, "features", None) else None
    if feature is not None and hasattr(feature, "int2str"):
        try:
            return str(feature.int2str(int(label_id)))
        except (TypeError, ValueError):
            pass
    return str(label_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-per-class", type=int, default=250,
                        help="Maximum photos to save per classification label (default: 250).")
    parser.add_argument("--output", type=Path, default=Path("dataset/real_world_eyewear_photos"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow additional files in an existing output directory; existing files are never replaced.")
    args = parser.parse_args()
    if args.max_per_class < 1:
        parser.error("--max-per-class must be at least 1")

    try:
        from datasets import load_dataset
    except ImportError:
        parser.error("Install optional dependencies first: pip install datasets pillow")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{args.split}_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        parser.error(f"{manifest_path} already exists. Choose another output folder or pass --overwrite.")

    try:
        dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
    except Exception as exc:
        parser.error(f"Could not open the public Hugging Face dataset/split: {exc}")

    counts: dict[str, int] = {}
    seen_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in dataset:
        label = _label_name(dataset, row.get("label", "unknown"))
        if counts.get(label, 0) >= args.max_per_class:
            if len(counts) >= 2 and all(value >= args.max_per_class for value in counts.values()):
                break
            continue
        try:
            image = _decode_image(row.get("image"))
        except Exception:
            continue

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        safe_label = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "unknown"
        image_name = f"{digest[:16]}.jpg"
        relative = Path("images") / args.split / safe_label / image_name
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        target.write_bytes(payload)
        counts[label] = counts.get(label, 0) + 1
        rows.append({
            "image_path": relative.as_posix(),
            "source_split": args.split,
            "source_label": label,
            "sha256": digest,
            "annotation_status": "UNLABELLED",
            "source_dataset": DATASET_ID,
            "source_license": "CC-BY-4.0",
        })
        if len(counts) >= 2 and all(value >= args.max_per_class for value in counts.values()):
            break

    if not rows:
        parser.error("No images were downloaded. Check dataset access and the requested split.")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    attribution = output / "SOURCE_ATTRIBUTION.txt"
    attribution.write_text(
        "Source: Eyeglasses vs Sunglasses Dataset\n"
        f"Dataset: {DATASET_URL}\n"
        "License declared by source: CC-BY-4.0\n"
        "Purpose: real product-image pool for manual eyewear part annotation.\n"
        "Important: source labels are product category labels, NOT segmentation masks.\n"
        "Do not use these images for YOLO segmentation training until part masks are\n"
        "created and human-reviewed. Keep the source split unchanged.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} photos to {output}")
    print(f"Per-label counts: {counts}")
    print(f"Manifest: {manifest_path}")
    print("Images are UNLABELLED for segmentation. No YOLO labels were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
