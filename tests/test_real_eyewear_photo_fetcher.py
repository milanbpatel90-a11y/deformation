from __future__ import annotations

import sys
import types

import pytest

from scripts.fetch_real_eyewear_photos import (
    DATASET_ID,
    _decode_image,
    _hub_file_location,
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


def test_hf_uri_is_converted_to_repo_relative_filename_and_pinned_revision():
    source_path = (
        "hf://datasets/shravya11/eyeglasses-dataset@7011dd3ac628040a8885a4b597399409643a9e04/"
        "train/eyeglasses/0/product/mt6960_0.jpg"
    )
    assert _hub_file_location(source_path) == (
        "train/eyeglasses/0/product/mt6960_0.jpg",
        "7011dd3ac628040a8885a4b597399409643a9e04",
    )


def test_plain_repo_relative_path_has_no_pinned_revision():
    assert _hub_file_location("train/eyeglasses/0/product/mt6960_0.jpg") == (
        "train/eyeglasses/0/product/mt6960_0.jpg",
        None,
    )


def test_incomplete_hf_directory_uri_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        _hub_file_location(
            "hf://datasets/shravya11/eyeglasses-dataset@7011dd3ac628040a8885a4b597399409643a9e04/train/eyeglasses/0/product"
        )


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

    hub_path = (
        "hf://datasets/shravya11/eyeglasses-dataset@7011dd3ac628040a8885a4b597399409643a9e04/"
        "train/eyeglasses/118/product/frame_0.jpg"
    )
    result = _decode_image({"path": hub_path, "bytes": None})

    assert result == "decoded-image"
    assert requested["download"] == {
        "repo_id": DATASET_ID,
        "filename": "train/eyeglasses/118/product/frame_0.jpg",
        "repo_type": "dataset",
        "revision": "7011dd3ac628040a8885a4b597399409643a9e04",
    }
    assert requested["opened"] == "C:/cache/frame_0.jpg"
