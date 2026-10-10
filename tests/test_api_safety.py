"""Regression tests for bounded upload handling."""

import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.api import safety


def _upload(contents: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(contents), filename="orbit.mp4")


def test_video_upload_is_streamed_to_disk(tmp_path) -> None:
    destination = tmp_path / "orbit.mp4"

    size = asyncio.run(safety.save_video_upload(_upload(b"video-content"), destination))

    assert size == len(b"video-content")
    assert destination.read_bytes() == b"video-content"


def test_empty_video_upload_is_rejected_and_removed(tmp_path) -> None:
    destination = tmp_path / "orbit.mp4"

    with pytest.raises(HTTPException) as raised:
        asyncio.run(safety.save_video_upload(_upload(b""), destination))

    assert raised.value.status_code == 400
    assert not destination.exists()


def test_oversized_video_upload_is_rejected_and_partial_file_removed(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "orbit.mp4"
    monkeypatch.setattr(safety, "MAX_VIDEO_BYTES", 4)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(safety.save_video_upload(_upload(b"12345"), destination))

    assert raised.value.status_code == 413
    assert not destination.exists()
