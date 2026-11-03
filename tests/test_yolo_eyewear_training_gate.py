from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from scripts.train_yolo_eyewear_parts import (
    EXPECTED_NAMES,
    parse_polygon_line,
    validate_dataset,
)


def _make_six_class_dataset(root: Path, status="HUMAN_VERIFIED_CORRECTED", shared_group=False):
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_name = f"{split}_001.jpg"
        (image_dir / image_name).write_bytes(b"fake image bytes")
        # One polygon for each of the six classes.
        lines = [
            f"{class_id} 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
            for class_id in range(6)
        ]
        (label_dir / f"{split}_001.txt").write_text("\n".join(lines), encoding="utf-8")

    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(EXPECTED_NAMES)},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = root / "review_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "local_image", "product_id", "annotation_status"],
        )
        writer.writeheader()
        for split in ("train", "val"):
            writer.writerow({
                "split": split,
                "local_image": f"images/{split}/{split}_001.jpg",
                "product_id": "same-product" if shared_group else f"product-{split}",
                "annotation_status": status,
            })
    return data_yaml, manifest


def test_polygon_parser_accepts_normalized_six_class_shape():
    assert parse_polygon_line("5 0.1 0.1 0.9 0.1 0.9 0.9", 1) == 5


@pytest.mark.parametrize(
    "line",
    [
        "0 0.1 0.1 1.2 0.1 0.9 0.9",
        "6 0.1 0.1 0.9 0.1 0.9 0.9",
        "2 0.1 0.1 0.9 0.1",
        "x 0.1 0.1 0.9 0.1 0.9 0.9",
    ],
)
def test_polygon_parser_rejects_invalid_lines(line):
    with pytest.raises(ValueError):
        parse_polygon_line(line, 1)


def test_training_gate_accepts_human_reviewed_six_class_dataset(tmp_path):
    data_yaml, manifest = _make_six_class_dataset(tmp_path)
    report = validate_dataset(data_yaml, manifest)
    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["val"]["images"] == 1
    assert report["class_names"] == EXPECTED_NAMES


def test_training_gate_rejects_unreviewed_ai_prelabels(tmp_path):
    data_yaml, manifest = _make_six_class_dataset(tmp_path, status="AI_PRELABEL_NOT_HUMAN_VERIFIED")
    with pytest.raises(ValueError, match="Refusing to train on unreviewed image"):
        validate_dataset(data_yaml, manifest)


def test_training_gate_rejects_product_group_leakage(tmp_path):
    data_yaml, manifest = _make_six_class_dataset(tmp_path, shared_group=True)
    with pytest.raises(ValueError, match="Product groups appear in multiple splits"):
        validate_dataset(data_yaml, manifest)


def test_training_gate_rejects_one_class_silhouette_yaml(tmp_path):
    data_yaml, manifest = _make_six_class_dataset(tmp_path)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config["names"] = {0: "eyewear"}
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Six-part class contract mismatch"):
        validate_dataset(data_yaml, manifest)
