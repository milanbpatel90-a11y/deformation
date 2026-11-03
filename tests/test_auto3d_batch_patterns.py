from pathlib import Path

from backend.pipeline.auto_3d_pipeline import Auto3DPipeline


def _pipeline_with_stubbed_processor():
    pipeline = object.__new__(Auto3DPipeline)
    pipeline.process_image = lambda image_path, output_dir: {"input_image": str(image_path)}
    return pipeline


def test_explicit_jpg_pattern_does_not_include_other_image_types(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "one.jpg").write_bytes(b"jpg")
    (image_dir / "two.png").write_bytes(b"png")
    (image_dir / "three.webp").write_bytes(b"webp")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    results = _pipeline_with_stubbed_processor().process_batch(
        image_dir, tmp_path / "output", pattern="*.jpg"
    )
    assert [Path(result["input_image"]).name for result in results] == ["one.jpg"]


def test_default_wildcard_accepts_supported_image_formats_only(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("one.jpg", "two.jpeg", "three.png", "four.webp", "five.bmp", "notes.txt"):
        (image_dir / name).write_bytes(b"image")

    results = _pipeline_with_stubbed_processor().process_batch(
        image_dir, tmp_path / "output"
    )
    assert [Path(result["input_image"]).name for result in results] == [
        "five.bmp", "four.webp", "one.jpg", "three.png", "two.jpeg"
    ]


def test_batch_creates_output_directory_and_writes_summary(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "one.jpg").write_bytes(b"jpg")
    output_dir = tmp_path / "nested" / "output"

    results = _pipeline_with_stubbed_processor().process_batch(image_dir, output_dir)

    assert len(results) == 1
    assert (output_dir / "batch_summary.json").is_file()


def test_batch_writes_summary_when_no_images_match(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "output"

    results = _pipeline_with_stubbed_processor().process_batch(image_dir, output_dir)

    assert results == []
    assert (output_dir / "batch_summary.json").is_file()


def test_batch_rejects_missing_input_directory(tmp_path):
    import pytest

    with pytest.raises(NotADirectoryError, match="Image input directory does not exist"):
        _pipeline_with_stubbed_processor().process_batch(
            tmp_path / "missing-images", tmp_path / "output"
        )
