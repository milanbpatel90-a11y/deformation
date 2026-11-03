from __future__ import annotations

import sys
import types

import pytest

from scripts.fetch_real_eyewear_photos import (
    DATASET_ID,
    _decode_image,
    _product_id_from_source_path,
    _source_path,
)


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        ("train/eyeglasses/118/product/frame_0.jpg", "118"),
        ("validation/sunglasses/369/product/frame_2.webp", "369"),
        (r"train\sunglasses\119\product\frame_1.jpg", "119"),
    ],
)
def test_product_id_is_extracted_from_product_folder(source_path, expected):
    assert _product_id_from_source_path(source_path) == expected


@pytest.mark.parametrize(
    "source_path",
    [
        "train/eyeglasses/118/review/review_0.jpg",
        "train/eyeglasses/118/product",
        None,
        "",
    ],
)
def test_review_or_incomplete_paths_are_not_catalogue_products(source_path):
    assert _product_id_from_source_path(source_path) is None


def test_source_path_is_read_from_hugging_face_image_record():
    record = {"path": "train/eyeglasses/118/product/frame_0.jpg", "bytes": None}
    assert _source_path(record) == record["path"]


def test_remote_hub_image_path_is_downloaded_before_pillow_open(monkeypatch):
    requested = {}

    class FakeOpenedImage:
        def convert(self, mode):
            assert mode == "RGB"
            return "decoded-image"

    class FakeImage:
        @staticmethod
        def open(path):
            requested["opened"] = path
            return FakeOpenedImage()

    def fake_hf_hub_download(**kwargs):
        requested["download"] = kwargs
        return "C:/cache/frame_0.jpg"

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImage
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = _decode_image(
        {"path": "train/eyeglasses/118/product/frame_0.jpg", "bytes": None}
    )

    assert result == "decoded-image"
    assert requested["download"] == {
        "repo_id": DATASET_ID,
        "filename": "train/eyeglasses/118/product/frame_0.jpg",
        "repo_type": "dataset",
    }
    assert requested["opened"] == "C:/cache/frame_0.jpg"
