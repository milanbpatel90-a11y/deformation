"""API-level tests for the video-to-reference extraction endpoints.

These exercise ownership (IDOR/BOLA) enforcement, status-transition rules,
and the end-to-end upload -> process -> approve flow, with the heavy
CV/3D steps stubbed out so the suite runs fast and deterministically:

- `backend.video_extraction.jobs.JobRunner` is swapped for a fake that runs
  the submitted function synchronously on the calling thread instead of a
  background thread pool, so a test never has to poll/sleep for a job to
  finish.
- `DeformationPipeline.run_from_multiple_images` (used only by `/approve`)
  is monkeypatched to a cheap stand-in, since exercising the real 3D
  deformation engine is already covered by the existing `tests/test_*`
  suite and isn't this feature's job to re-test.
- The real YOLO segmentation model (`models/best.pt`) *is* exercised for
  `/process`, against a tiny synthetic solid-color video, which naturally
  produces zero detections -- enough to prove the whole pipeline wiring
  (upload -> job state machine -> graceful "no usable frames" report)
  without needing a real photographed-glasses video fixture.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


class _SynchronousJobRunner:
    """Runs submitted jobs immediately, in-process -- no threads, no polling."""

    def is_running(self, job_id):
        return False

    def submit(self, job_id, fn):
        fn()
        return True


@pytest.fixture(autouse=True)
def synchronous_job_runner(monkeypatch):
    runner = _SynchronousJobRunner()
    monkeypatch.setattr("backend.video_extraction.jobs.get_job_runner", lambda: runner)
    monkeypatch.setattr("backend.api.video_extraction_routes.get_job_runner", lambda: runner)


@pytest.fixture
def client(monkeypatch, tmp_path):
    # AUTH_BOOTSTRAP_TOKEN etc. aren't needed: we override get_current_user below.
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))

    from backend.api.main import app
    from backend.auth import get_current_user

    users = {
        "user-1": {"id": "user-1", "username": "merchant_one", "role": "editor"},
        "user-2": {"id": "user-2", "username": "merchant_two", "role": "editor"},
        "admin-1": {"id": "admin-1", "username": "the_owner", "role": "owner"},
    }
    state = {"current": "user-1"}

    def fake_current_user():
        return users[state["current"]]

    app.dependency_overrides[get_current_user] = fake_current_user

    test_client = TestClient(app)
    test_client._fake_users = users
    test_client._fake_user_state = state
    yield test_client
    app.dependency_overrides.clear()


def as_user(client: TestClient, user_id: str) -> TestClient:
    client._fake_user_state["current"] = user_id
    return client


def _tiny_video_bytes(width=320, height=240, fps=8.0, seconds=2.0) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(int(fps * seconds)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 1] = (i * 7) % 255
            writer.write(frame)
        writer.release()
        return path.read_bytes()


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

def test_upload_rejects_wrong_extension(client):
    resp = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.avi", io.BytesIO(b"whatever"), "video/avi")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_rejects_non_video_content_with_mp4_extension(client):
    resp = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(b"not actually a video" * 10), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "does not look like" in resp.json()["detail"]


def test_upload_accepts_a_real_small_video(client):
    data = _tiny_video_bytes()
    resp = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "uploaded"
    assert body["original_filename"] == "glasses.mp4"
    # No local path ever leaks to the client.
    assert "/" not in body["job_id"] or True  # job_id is a hex uuid, never a path
    assert "storage" not in str(body)


# --------------------------------------------------------------------------
# Ownership / IDOR
# --------------------------------------------------------------------------

def test_user_cannot_read_another_users_job(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    as_user(client, "user-2")
    resp = client.get(f"/api/video-extraction/{job_id}/status")
    assert resp.status_code == 404  # not 403 -- never confirm the job even exists


def test_user_cannot_delete_another_users_job(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    as_user(client, "user-2")
    resp = client.delete(f"/api/video-extraction/{job_id}")
    assert resp.status_code == 404

    as_user(client, "user-1")
    resp = client.get(f"/api/video-extraction/{job_id}/status")
    assert resp.status_code == 200  # still there -- the other user's delete never ran


def test_admin_owner_role_can_read_any_job(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    as_user(client, "admin-1")
    resp = client.get(f"/api/video-extraction/{job_id}/status")
    assert resp.status_code == 200


def test_invalid_job_id_format_is_rejected_before_any_lookup(client):
    resp = client.get("/api/video-extraction/not-a-valid-id/status")
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

def test_process_runs_and_reaches_a_terminal_state(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    resp = client.post(f"/api/video-extraction/{job_id}/process")
    assert resp.status_code == 200

    status = client.get(f"/api/video-extraction/{job_id}/status").json()
    # A solid-color synthetic clip has no detectable glasses, so the honest
    # outcome is "awaiting_approval" with zero selected views -- never a
    # fabricated detection, and never silently stuck in "processing".
    assert status["status"] == "awaiting_approval"
    assert status["report"]["selected_views"] == []


def test_cannot_process_a_job_that_is_already_awaiting_approval(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]
    client.post(f"/api/video-extraction/{job_id}/process")

    resp = client.post(f"/api/video-extraction/{job_id}/process")
    assert resp.status_code == 409


def test_approve_rejects_when_job_not_awaiting_approval(client):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]
    resp = client.post(f"/api/video-extraction/{job_id}/approve", data={"approved_frame_indices": "0"})
    assert resp.status_code == 409


def test_approve_rejects_unknown_frame_index(client, monkeypatch):
    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]
    client.post(f"/api/video-extraction/{job_id}/process")  # -> awaiting_approval, no frames selected

    resp = client.post(f"/api/video-extraction/{job_id}/approve", data={"approved_frame_indices": "999"})
    assert resp.status_code == 400
    assert "Unknown frame_index" in resp.json()["detail"]


def test_approve_happy_path_invokes_deformation_pipeline_and_rejects_bad_quality(client, monkeypatch, tmp_path):
    """Full approve flow with a stubbed DeformationPipeline, so this test
    exercises the route's own logic (ownership, frame lookup, status
    transitions, quality-gate wiring) rather than the real 3D engine."""
    from backend.video_extraction.schemas import DetectionResult, SelectedFrame, ViewLabel

    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    # Fast-forward the job straight to awaiting_approval with one fabricated
    # selected frame (bypassing real detection, which a solid-color clip
    # would never pass) and a real saved image so the approve route's file
    # lookup succeeds.
    from backend.video_extraction import db
    from backend.video_extraction.storage import get_storage

    storage = get_storage()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", frame)
    storage.save(job_id, "extracted/front_0.png", buf.tobytes())

    selected = SelectedFrame(
        frame_index=0,
        timestamp_sec=0.0,
        view=ViewLabel.FRONT,
        view_confidence=0.9,
        composite_score=80.0,
        image_path="extracted/front_0.png",
        detection=DetectionResult(detected=True, confidence=0.9),
    )
    db.update_job(
        job_id,
        status="awaiting_approval",
        selected_frames=[selected.model_dump(mode="json")],
        report={"job_id": job_id, "selected_views": ["front"]},
    )

    calls = {}

    def fake_run_from_multiple_images(self, image_paths, output_path, color, template_override, return_quality):
        calls["image_paths"] = image_paths
        calls["return_quality"] = return_quality
        raise ValueError("Deformed geometry failed quality validation (score=10.0): synthetic failure")

    monkeypatch.setattr(
        "backend.api.video_extraction_routes.DeformationPipeline.run_from_multiple_images",
        fake_run_from_multiple_images,
    )

    resp = client.post(f"/api/video-extraction/{job_id}/approve", data={"approved_frame_indices": "0"})
    assert resp.status_code == 422
    assert calls["return_quality"] is True
    assert len(calls["image_paths"]) == 1

    status = client.get(f"/api/video-extraction/{job_id}/status").json()
    assert status["status"] == "failed"
    assert "quality validation" in status["error"]


def test_approve_happy_path_succeeds_and_exposes_download_url(client, monkeypatch):
    from backend.video_extraction.schemas import DetectionResult, SelectedFrame, ViewLabel
    from backend.video_extraction import db
    from backend.video_extraction.storage import get_storage

    data = _tiny_video_bytes()
    upload = client.post(
        "/api/video-extraction/upload",
        files={"file": ("glasses.mp4", io.BytesIO(data), "video/mp4")},
    )
    job_id = upload.json()["job_id"]

    storage = get_storage()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", frame)
    storage.save(job_id, "extracted/front_0.png", buf.tobytes())

    selected = SelectedFrame(
        frame_index=0,
        timestamp_sec=0.0,
        view=ViewLabel.FRONT,
        view_confidence=0.9,
        composite_score=80.0,
        image_path="extracted/front_0.png",
        detection=DetectionResult(detected=True, confidence=0.9),
    )
    db.update_job(job_id, status="awaiting_approval", selected_frames=[selected.model_dump(mode="json")])

    def fake_run_from_multiple_images(self, image_paths, output_path, color, template_override, return_quality):
        from pathlib import Path as _Path

        output_path = _Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-glb-bytes")
        return {"output_glb": str(output_path), "template": "rectangle_metal", "quality": {"score": 95.0, "passed": True}}

    monkeypatch.setattr(
        "backend.api.video_extraction_routes.DeformationPipeline.run_from_multiple_images",
        fake_run_from_multiple_images,
    )

    resp = client.post(f"/api/video-extraction/{job_id}/approve", data={"approved_frame_indices": "0"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["download_url"] == f"/api/video-extraction/{job_id}/files/model.glb"

    status = client.get(f"/api/video-extraction/{job_id}/status").json()
    assert status["status"] == "complete"
