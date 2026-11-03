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
from pathlib import Path, PurePosixPath
from typing import Any


DATASET_ID = "shravya11/eyeglasses-dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"


def _source_path(value: Any) -> str | None:
    """Return the source repository path carried by an Image(decode=False) value."""
    if isinstance(value, dict):
        path = value.get("path")
        return str(path).replace("\\", "/") if path else None
    if isinstance(value, (str, Path)):
        return str(value).replace("\\", "/")
    filename = getattr(value, "filename", None)
    return str(filename).replace("\\", "/") if filename else None


def _product_id_from_source_path(source_path: str | None) -> str | None:
    """Extract the product grouping ID from .../<product_id>/product/<image>."""
    if not source_path:
        return None
    parts = [part for part in PurePosixPath(source_path.replace("\\", "/")).parts if part not in ("", "/")]
    lowered = [part.lower() for part in parts]
    try:
        product_folder_index = lowered.index("product")
    except ValueError:
        return None
    # The source layout is split/category/product_id/product/image.
    if product_folder_index < 3 or product_folder_index + 1 >= len(parts):
        return None
    return parts[product_folder_index - 1]


def _decode_image(value: Any, source_path: str | None = None):
    """Decode an image whether the stream supplies bytes, a local path or a Hub path."""
    from PIL import Image

    if hasattr(value, "convert"):
        return value.convert("RGB")

    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        source_path = _source_path(value) or source_path
    elif isinstance(value, (str, Path)):
        source_path = str(value)

    if source_path:
        normalized_path = source_path.replace("\\", "/").lstrip("/")
        local_path = Path(source_path)
        if local_path.is_file():
            return Image.open(local_path).convert("RGB")

        # With a streaming Image(decode=False) feature, path can refer to a
        # file inside the remote dataset repository, not a local disk file.
        # Resolve it through the Hub cache before asking Pillow to open it.
        from huggingface_hub import hf_hub_download

        cached_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=normalized_path,
            repo_type="dataset",
        )
        return Image.open(cached_path).convert("RGB")

    raise ValueError("Dataset row has no image bytes or source path.")


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
    args = parser.parse_args()
    if args.max_per_class < 1:
        parser.error("--max-per-class must be at least 1")

    try:
        from datasets import Image, load_dataset
    except ImportError:
        parser.error("Install optional dependencies first: pip install datasets pillow")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{args.split}_manifest.csv"
    if manifest_path.exists():
        parser.error(f"{manifest_path} already exists. Choose a new output folder to preserve the previous manifest.")

    try:
        dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
        # Keep each source path so product-model folder IDs can be retained
        # in the annotation manifest. Remote paths are resolved by _decode_image.
        dataset = dataset.cast_column("image", Image(decode=False))
    except Exception as exc:
        parser.error(f"Could not open the public Hugging Face dataset/split: {exc}")

    counts: dict[str, int] = {}
    seen_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    scanned = 0
    skipped_non_product = 0
    skipped_missing_path = 0
    decode_failures = 0
    sample_decode_errors: list[str] = []

    for row in dataset:
        scanned += 1
        label = _label_name(dataset, row.get("label", "unknown"))
        if counts.get(label, 0) >= args.max_per_class:
            if len(counts) >= 2 and all(value >= args.max_per_class for value in counts.values()):
                break
            continue

        source_image = row.get("image")
        source_path = _source_path(source_image)
        product_id = _product_id_from_source_path(source_path)
        if not source_path:
            skipped_missing_path += 1
            continue
        # Do not save review/customer photos. Only use catalogue product shots.
        if not product_id:
            skipped_non_product += 1
            continue

        try:
            image = _decode_image(source_image, source_path)
        except Exception as exc:
            decode_failures += 1
            if len(sample_decode_errors) < 3:
                sample_decode_errors.append(f"{source_path}: {type(exc).__name__}: {exc}")
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
            # A previous run of the older script may have left the exact image
            # on disk before it failed to write a manifest. Reuse it without
            # overwriting any local file, then rebuild its manifest row.
            existing_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if existing_digest != digest:
                continue
        else:
            target.write_bytes(payload)

        training_split = "val" if args.split == "validation" else "train"
        rows.append({
            "split": training_split,
            "local_image": relative.as_posix(),
            "product_id": product_id,
            "source_split": args.split,
            "source_label": label,
            "sha256": digest,
            "annotation_status": "UNLABELLED",
            "reviewer": "",
            "source_dataset": DATASET_ID,
            "source_license": "CC-BY-4.0",
        })
        counts[label] = counts.get(label, 0) + 1

        if len(counts) >= 2 and all(value >= args.max_per_class for value in counts.values()):
            break

    if not rows:
        details = (
            f"Scanned {scanned} rows; saved 0 catalogue product images. "
            f"Missing source paths: {skipped_missing_path}; "
            f"non-product/review paths: {skipped_non_product}; "
            f"image decode/download failures: {decode_failures}."
        )
        if sample_decode_errors:
            details += " First decode errors: " + " | ".join(sample_decode_errors)
        parser.error(details)

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
    if decode_failures:
        print(f"Warning: {decode_failures} image rows could not be decoded/downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
