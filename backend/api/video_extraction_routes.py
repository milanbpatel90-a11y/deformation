"""FastAPI routes for the video-to-reference extraction feature.

Upload a video of physical eyeglasses -> extract/segment/quality-filter
candidate frames -> classify by viewing angle -> let the merchant review
and approve the best reference images -> hand the approved references to
the existing, UNMODIFIED `DeformationPipeline` to generate and validate a
GLB, exactly as the task brief's acceptance criteria describe.

Ownership model: this app has no multi-tenant "merchant" table -- accounts
are `backend.auth`'s existing owner/editor/operator/viewer roles. Every job
is tied to the uploading user's `owner_user_id`; every route below re-checks
that the caller either owns the job or holds the `owner` (admin) role before
touching it, so one account can never read, approve or delete another
account's job (IDOR/BOLA protection -- see `_authorize_job`). All mutating
routes additionally require the same `generate_models` permission tier
(`owner`/`editor`/`operator`) the existing `/api/deform*` endpoints use;
`viewer` accounts get read-only access to their own jobs.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.auth import require_roles
from backend.pipeline import DeformationPipeline
from backend.video_extraction import db
from backend.video_extraction.config import get_settings
from backend.video_extraction.exceptions import InvalidVideoError, StorageError, VideoExtractionError
from backend.video_extraction.extraction_pipeline import VideoExtractionPipeline
from backend.video_extraction.jobs import get_job_runner
from backend.video_extraction.schemas import CalibrationInput
from backend.video_extraction.storage import get_storage
from backend.video_extraction.video_processor import (
    safe_temp_video_name,
    validate_filename_extension,
    validate_magic_bytes,
    validate_upload_size,
)

logger = logging.getLogger("video_extraction.api")

router = APIRouter(prefix="/api/video-extraction", tags=["video-extraction"])

_CAN_WRITE = require_roles("owner", "editor", "operator")
_CAN_READ = require_roles("owner", "editor", "operator", "viewer")

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _new_job_id() -> str:
    return uuid.uuid4().hex


def _validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid job id.")
    return job_id


def _authorize_job(job_id: str, user: dict) -> dict:
    """Load a job and enforce ownership (IDOR/BOLA protection).

    Any authenticated role can read/act on their OWN job. Only the `owner`
    (admin) role may act on a job it does not own -- matching the "Admin
    access according to existing role architecture" requirement.
    """
    _validate_job_id(job_id)
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["owner_user_id"] != user["id"] and user["role"] != "owner":
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _public_job(job: dict) -> dict:
    """Never leak local filesystem paths -- only relative/derived fields."""
    return {
        "job_id": job["id"],
        "status": job["status"],
        "original_filename": job["original_filename"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "progress_message": job.get("progress_message"),
        "error": _public_error(job.get("error")),
        "report": job.get("report"),
        "calibration": job.get("calibration"),
    }


def _public_error(error: str | None) -> str | None:
    """Strip the internal traceback (kept in the DB for operators/logs) down
    to a clean, client-safe first line -- never return a stack trace or a
    filesystem path to the caller."""
    if not error:
        return None
    return error.splitlines()[0][:500]


def _public_frame(job_id: str, frame: dict) -> dict:
    storage = get_storage()
    return {
        "frame_index": frame["frame_index"],
        "timestamp_sec": frame["timestamp_sec"],
        "view": frame["view"],
        "view_confidence": frame["view_confidence"],
        "composite_score": frame["composite_score"],
        "approved": frame.get("approved", False),
        "manually_replaced": frame.get("manually_replaced", False),
        "detection": frame["detection"],
        "image_url": storage.local_url(job_id, frame["image_path"]),
        "overlay_url": storage.local_url(job_id, frame["overlay_path"]) if frame.get("overlay_path") else None,
        "cutout_url": storage.local_url(job_id, frame["cutout_path"]) if frame.get("cutout_path") else None,
        "mask_urls": {k: storage.local_url(job_id, v) for k, v in frame.get("mask_paths", {}).items()},
    }


@router.post("/upload", status_code=201)
async def upload_video(
    file: UploadFile = File(..., description="MP4 or MOV video of the physical glasses"),
    user: dict = Depends(_CAN_WRITE),
):
    """Validate and store an uploaded video; does not start processing yet."""
    settings = get_settings()

    active = len([j for j in db.list_jobs(user["id"]) if j["status"] in {"uploaded", "queued", "processing"}])
    if active >= settings.max_jobs_per_user:
        raise HTTPException(
            status_code=429,
            detail=f"You already have {active} video-extraction job(s) in progress. "
            "Wait for one to finish before starting another.",
        )

    # Never trust the filename beyond its extension.
    try:
        validate_filename_extension(file.filename or "", settings)
    except InvalidVideoError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Stream the upload to a temp buffer in bounded chunks instead of
    # `await file.read()` on the whole thing, so a malformed/huge upload
    # can't exhaust memory before the size check below even runs.
    chunk_size = 4 * 1024 * 1024
    data = bytearray()
    too_large = False
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > settings.max_upload_bytes:
            too_large = True
            break
    if too_large:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {settings.max_upload_mb:.0f} MB limit.",
        )
    try:
        validate_upload_size(len(data), settings)
        validate_magic_bytes(bytes(data[:256]))
    except InvalidVideoError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = _new_job_id()
    storage = get_storage()
    ext = Path(file.filename or "").suffix.lower()
    relpath = safe_temp_video_name(job_id, ext)
    try:
        storage.save(job_id, relpath, bytes(data))
    except StorageError as exc:
        raise HTTPException(status_code=500, detail="Could not store the uploaded video.") from exc

    job = db.create_job(job_id, user["id"], original_filename=Path(file.filename or "video").name)
    db.update_job(job_id, video_relpath=relpath)
    return _public_job(job)


@router.post("/{job_id}/process")
def process_video(
    job_id: str,
    known_frame_width_mm: float | None = Form(None, ge=80.0, le=200.0),
    user: dict = Depends(_CAN_WRITE),
):
    """Kick off background extraction for an uploaded video."""
    job = _authorize_job(job_id, user)
    if job["status"] not in {"uploaded", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job['status']}'; only an 'uploaded' or 'failed' job can be (re)processed. "
            "Use /regenerate once it has reached awaiting_approval.",
        )

    storage = get_storage()
    video_path = storage.job_dir(job_id) / job["video_relpath"]
    calibration = (
        CalibrationInput(known_frame_width_mm=known_frame_width_mm) if known_frame_width_mm else None
    )
    db.update_job(
        job_id,
        status="queued",
        progress_message="Queued for processing.",
        error=None,
        calibration=calibration.model_dump() if calibration else None,
    )

    runner = get_job_runner()
    submitted = runner.submit(job_id, lambda: _run_extraction(job_id, video_path, calibration))
    if not submitted:
        raise HTTPException(status_code=409, detail="This job is already being processed.")
    return {"message": "Processing started.", "job_id": job_id, "status": "queued"}


def _run_extraction(job_id: str, video_path: Path, calibration: CalibrationInput | None) -> None:
    storage = get_storage()
    db.update_job(job_id, status="processing", progress_message="Starting...")
    pipeline = VideoExtractionPipeline()

    def progress(message: str) -> None:
        db.update_job(job_id, progress_message=message)

    try:
        result = pipeline.run(job_id, video_path, storage, calibration=calibration, progress_cb=progress)
    except VideoExtractionError as exc:
        db.update_job(job_id, status="failed", error=str(exc))
        return

    db.update_job(
        job_id,
        status="awaiting_approval",
        progress_message="Extraction complete. Awaiting merchant review.",
        report=result.report.model_dump(mode="json"),
        selected_frames=[f.model_dump(mode="json") for f in result.selected_frames],
        measurements=result.measurements_bundle,
    )


@router.get("/{job_id}/status")
def job_status(job_id: str, user: dict = Depends(_CAN_READ)):
    job = _authorize_job(job_id, user)
    return _public_job(job)


@router.get("/{job_id}/results")
def job_results(job_id: str, user: dict = Depends(_CAN_READ)):
    job = _authorize_job(job_id, user)
    if job["status"] in {"uploaded", "queued", "processing"}:
        raise HTTPException(status_code=409, detail=f"Job is still '{job['status']}'; no results yet.")
    frames = job.get("selected_frames") or []
    return {
        "job_id": job_id,
        "status": job["status"],
        "report": job.get("report"),
        "measurements": job.get("measurements"),
        "frames": [_public_frame(job_id, f) for f in frames],
        "deform_result": job.get("deform_result"),
    }


@router.get("/{job_id}/frames")
def job_frames(job_id: str, user: dict = Depends(_CAN_READ)):
    job = _authorize_job(job_id, user)
    frames = job.get("selected_frames") or []
    return {"job_id": job_id, "frames": [_public_frame(job_id, f) for f in frames]}


_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_\-./]+$")


@router.get("/{job_id}/files/{relative_path:path}")
def job_file(job_id: str, relative_path: str, user: dict = Depends(_CAN_READ)):
    """Serve one stored asset (extracted frame/mask/overlay/cutout/report)
    for a job the caller owns (or is an admin for). This is the private,
    authenticated/ownership-checked substitute for a public bucket URL (see
    `storage.py`'s docstring)."""
    _authorize_job(job_id, user)
    if not _SAFE_RELATIVE.fullmatch(relative_path):
        raise HTTPException(status_code=400, detail="Invalid path.")
    storage = get_storage()
    try:
        path = storage.open_path(job_id, relative_path)
    except StorageError:
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "application/json" if path.suffix == ".json" else "image/png" if path.suffix == ".png" else "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.post("/{job_id}/approve")
async def approve_frames(
    job_id: str,
    approved_frame_indices: str = Form(..., description="Comma-separated frame_index values to approve"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(_CAN_WRITE),
):
    """Merchant approves a subset of extracted reference frames, then the
    approved images are handed to the existing, untouched
    `DeformationPipeline.run_from_multiple_images` to generate + validate a
    GLB. Nothing is auto-approved: the caller must explicitly list which
    frame_index values to use."""
    job = _authorize_job(job_id, user)
    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job['status']}'; approval requires 'awaiting_approval'.",
        )

    try:
        wanted = {int(x) for x in approved_frame_indices.split(",") if x.strip() != ""}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="approved_frame_indices must be a comma-separated list of integers.") from exc
    if not wanted:
        raise HTTPException(status_code=400, detail="At least one frame_index must be approved.")

    frames = job.get("selected_frames") or []
    frames_by_index = {f["frame_index"]: f for f in frames}
    missing = wanted - frames_by_index.keys()
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown frame_index value(s): {sorted(missing)}")

    storage = get_storage()
    for frame in frames:
        frame["approved"] = frame["frame_index"] in wanted
    db.update_job(job_id, selected_frames=frames, status="generating", progress_message="Generating 3D model...")

    approved_paths = [storage.open_path(job_id, frames_by_index[i]["image_path"]) for i in sorted(wanted)]

    try:
        pipeline = DeformationPipeline()
        out_path = storage.job_dir(job_id) / "model.glb"
        result = pipeline.run_from_multiple_images(
            approved_paths,
            out_path,
            color=color,
            template_override=template,
            return_quality=True,
        )
    except ValueError as exc:
        db.update_job(job_id, status="failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        logger.exception("GLB generation failed for video-extraction job %s", job_id)
        db.update_job(job_id, status="failed", error=f"{exc.__class__.__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Model generation failed. See server logs for details.") from exc

    result["download_url"] = storage.local_url(job_id, "model.glb")
    db.update_job(job_id, status="complete", progress_message="Model ready.", deform_result=result, selected_frames=frames)
    return {"message": "Approved and generated.", "job_id": job_id, "result": result}


@router.post("/{job_id}/regenerate")
def regenerate(job_id: str, user: dict = Depends(_CAN_WRITE)):
    """Re-run extraction from the already-uploaded video (e.g. after
    tweaking calibration), discarding any previous selection/approval."""
    job = _authorize_job(job_id, user)
    if job["status"] not in {"awaiting_approval", "complete", "failed"}:
        raise HTTPException(status_code=409, detail=f"Cannot regenerate while job is '{job['status']}'.")

    storage = get_storage()
    video_path = storage.job_dir(job_id) / job["video_relpath"]
    calibration = CalibrationInput(**job["calibration"]) if job.get("calibration") else None

    db.update_job(
        job_id,
        status="queued",
        progress_message="Re-queued for processing.",
        error=None,
        selected_frames=None,
        measurements=None,
        deform_result=None,
    )
    runner = get_job_runner()
    submitted = runner.submit(job_id, lambda: _run_extraction(job_id, video_path, calibration))
    if not submitted:
        raise HTTPException(status_code=409, detail="This job is already being processed.")
    return {"message": "Regeneration started.", "job_id": job_id, "status": "queued"}


@router.delete("/{job_id}")
def delete_job(job_id: str, user: dict = Depends(_CAN_WRITE)):
    job = _authorize_job(job_id, user)
    runner = get_job_runner()
    if runner.is_running(job_id):
        raise HTTPException(status_code=409, detail="Cannot delete a job while it is actively processing.")
    storage = get_storage()
    storage.delete(job_id)
    db.delete_job(job_id)
    return {"message": "Job deleted."}


@router.get("")
def list_jobs(user: dict = Depends(_CAN_READ)):
    """List the caller's own jobs (or, for `owner`/admin, still only their
    own -- use the admin users list for cross-account auditing)."""
    jobs = db.list_jobs(user["id"])
    return {"jobs": [_public_job(j) for j in jobs]}
