from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.inspect_real_eyewear_dataset import inspect_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "local_image", "product_id", "source_split", "source_label",
        "sha256", "annotation_status", "source_dataset", "source_license",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_inspection_checks_images_hashes_and_creates_unlabelled_review_queue(tmp_path):
    dataset = tmp_path / "dataset"
    train_image = dataset / "images/train/eyeglasses/a.jpg"
    val_image = dataset / "images/val/sunglasses/b.jpg"
    train_hash = _write_image(train_image, (30, 40, 50))
    val_hash = _write_image(val_image, (60, 70, 80))
    common = {
        "source_dataset": "example/dataset",
        "source_license": "CC-BY-4.0",
        "annotation_status": "UNLABELLED",
    }
    _write_manifest(dataset / "train_manifest.csv", [{
        **common, "split": "train", "source_split": "train",
        "local_image": "images/train/eyeglasses/a.jpg",
        "product_id": "p001", "source_label": "eyeglasses", "sha256": train_hash,
    }])
    _write_manifest(dataset / "validation_manifest.csv", [{
        **common, "split": "val", "source_split": "validation",
        "local_image": "images/val/sunglasses/b.jpg",
        "product_id": "p002", "source_label": "sunglasses", "sha256": val_hash,
    }])

    output = dataset / "inspection"
    report = inspect_dataset(dataset, output)

    assert report["images_in_manifests"] == 2
    assert report["images_decodable"] == 2
    assert report["sha256_matches"] == 2
    assert report["cross_split_duplicate_images"] == []
    assert report["cross_split_product_leakage"] == []
    assert report["contact_sheets"]
    assert (output / "inspection_report.json").is_file()
    with (output / "annotation_review_queue.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        queue = list(csv.DictReader(handle))
    assert len(queue) == 2
    assert {row["annotation_status"] for row in queue} == {"UNLABELLED"}
    assert {row["review_status"] for row in queue} == {"NOT_REVIEWED"}
    assert json.loads((output / "inspection_report.json").read_text())[
        "images_decodable"
    ] == 2


def test_inspection_detects_product_group_leakage_across_splits(tmp_path):
    dataset = tmp_path / "dataset"
    train_hash = _write_image(
        dataset / "images/train/eyeglasses/a.jpg", (10, 20, 30)
    )
    val_hash = _write_image(
        dataset / "images/val/eyeglasses/b.jpg", (40, 50, 60)
    )
    base = {
        "product_id": "same-product",
        "source_label": "eyeglasses",
        "source_dataset": "example/dataset",
        "source_license": "CC-BY-4.0",
        "annotation_status": "UNLABELLED",
    }
    _write_manifest(dataset / "train_manifest.csv", [{
        **base, "split": "train", "source_split": "train",
        "local_image": "images/train/eyeglasses/a.jpg", "sha256": train_hash,
    }])
    _write_manifest(dataset / "validation_manifest.csv", [{
        **base, "split": "val", "source_split": "validation",
        "local_image": "images/val/eyeglasses/b.jpg", "sha256": val_hash,
    }])

    report = inspect_dataset(dataset, dataset / "inspection")

    assert report["cross_split_product_leakage"] == [{
        "category": "eyeglasses",
        "product_id": "same-product",
        "splits": ["train", "val"],
    }]
