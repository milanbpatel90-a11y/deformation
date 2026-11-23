from __future__ import annotations

import pytest

from backend.video_extraction import db
from backend.video_extraction.exceptions import StorageError
from backend.video_extraction.storage import VideoExtractionStorage, get_storage


def test_save_and_open_round_trip(tmp_path):
    storage = VideoExtractionStorage(root=tmp_path)
    storage.save("job1", "extracted/front_0.png", b"fake-png-bytes")
    path = storage.open_path("job1", "extracted/front_0.png")
    assert path.read_bytes() == b"fake-png-bytes"


def test_path_traversal_in_relative_path_is_rejected(tmp_path):
    storage = VideoExtractionStorage(root=tmp_path)
    storage.save("job1", "extracted/front_0.png", b"data")
    with pytest.raises(StorageError):
        storage.open_path("job1", "../job2/secret.png")
    with pytest.raises(StorageError):
        storage.save("job1", "../escape.png", b"data")


def test_job_id_with_separators_is_rejected(tmp_path):
    storage = VideoExtractionStorage(root=tmp_path)
    with pytest.raises(StorageError):
        storage.save("../other_job", "x.png", b"data")
    with pytest.raises(StorageError):
        storage.save("job/with/slash", "x.png", b"data")


def test_delete_removes_entire_job_tree(tmp_path):
    storage = VideoExtractionStorage(root=tmp_path)
    storage.save("job1", "extracted/front_0.png", b"data")
    assert storage.exists("job1", "extracted/front_0.png")
    storage.delete("job1")
    assert not storage.exists("job1", "extracted/front_0.png")


def test_local_url_is_routed_through_authenticated_endpoint_not_a_bare_path(tmp_path):
    storage = VideoExtractionStorage(root=tmp_path)
    url = storage.local_url("job1", "extracted/front_0.png")
    assert url == "/api/video-extraction/job1/files/extracted/front_0.png"
    assert str(tmp_path) not in url


def test_job_crud_round_trip():
    job = db.create_job("job-abc", owner_user_id="user-1", original_filename="my video.mp4")
    assert job["status"] == "uploaded"
    assert job["owner_user_id"] == "user-1"

    updated = db.update_job("job-abc", status="processing", progress_message="working", report={"a": 1})
    assert updated["status"] == "processing"
    assert updated["progress_message"] == "working"
    assert updated["report"] == {"a": 1}

    fetched = db.get_job("job-abc")
    assert fetched["report"] == {"a": 1}

    db.delete_job("job-abc")
    assert db.get_job("job-abc") is None


def test_list_jobs_is_scoped_to_owner():
    db.create_job("job-a", owner_user_id="user-1", original_filename="a.mp4")
    db.create_job("job-b", owner_user_id="user-2", original_filename="b.mp4")

    mine = db.list_jobs("user-1")
    assert [j["id"] for j in mine] == ["job-a"]

    everyone = db.list_jobs(None)
    assert {j["id"] for j in everyone} == {"job-a", "job-b"}
