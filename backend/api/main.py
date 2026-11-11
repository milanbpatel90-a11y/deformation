"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image
from starlette.concurrency import run_in_threadpool

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from backend.auth import (
    SESSION_COOKIE_NAME,
    admin_router,
    auth_router,
    csrf_token_is_valid,
    get_current_user,
    get_current_user_optional,
    require_roles,
)
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.api import rim_detection_routes


logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = max(1, int(os.getenv("MAX_IMAGE_BYTES", 20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = max(1, int(os.getenv("MAX_IMAGE_PIXELS", 40_000_000)))
MAX_MULTI_VIEW_TOTAL_BYTES = max(1, int(os.getenv("MAX_MULTI_VIEW_TOTAL_BYTES", 100 * 1024 * 1024)))


async def _save_validated_image(upload: UploadFile, destination: Path) -> int:
    """Read a bounded upload, validate its format and dimensions, then persist it."""
    contents = await upload.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="An uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="An image exceeds the configured upload size limit.")
    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise HTTPException(status_code=400, detail="Use a JPEG, PNG, WebP, or BMP image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the processing limit.")
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="An uploaded file is not a valid supported image.") from exc
    destination.write_bytes(contents)
    return len(contents)


def _record_output_owner(metadata_path: Path, user: dict) -> None:
    if not metadata_path.is_file():
        raise RuntimeError("Generation metadata is missing; refusing to publish an unowned GLB.")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["created_by_user_id"] = user["id"]
    payload["created_by_username"] = user["username"]
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, metadata_path)


def _remove_failed_output(output_path: Path) -> None:
    for artifact in (output_path, output_path.with_suffix(".metadata.json")):
        try:
            artifact.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove partial generation artifact: %s", artifact)


app = FastAPI(
    title="Defirmation API",
    description="Eyewear template deformation and virtual try-on — protected with role-based accounts",
    version="1.2.0",
)

# The browser viewer is served by this same application. Cross-origin access is
# disabled unless explicitly allowlisted for a separate trusted frontend.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
    )


@app.middleware("http")
async def csrf_protection(request: Request, call_next):
    """Require the double-submit CSRF token on authenticated state changes."""
    safe_methods = {"GET", "HEAD", "OPTIONS"}
    exempt_paths = {"/api/auth/login", "/api/auth/setup"}
    if (
        request.method not in safe_methods
        and request.url.path not in exempt_paths
        and request.cookies.get(SESSION_COOKIE_NAME)
        and not csrf_token_is_valid(request)
    ):
        return JSONResponse(
            status_code=403,
            content={"detail": "CSRF validation failed. Refresh the page and try again."},
        )
    return await call_next(request)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PREFERRED_TEMPLATES = (
    "rectangle_metal",
    "cat_eye_gold",
    "round_wire_metal",
    "oversized_clear",
    "slim_rectangle_steel",
    "rectangle_plastic",
)

pipeline = DeformationPipeline()
library = TemplateLibrary()
viewer_dir = Path(__file__).resolve().parents[2] / "viewer"

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(rim_detection_routes.router)


def _pick_active_template(available: list[str]) -> str | None:
    for name in PREFERRED_TEMPLATES:
        if name in available:
            return name
    return available[0] if available else None


@app.get("/", include_in_schema=False)
def root(request: Request):
    """Send signed-in users to the viewer and everyone else to the login page."""
    return RedirectResponse(
        url="/viewer/" if get_current_user_optional(request) else "/login",
        status_code=303,
    )


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(viewer_dir / "login.html")


@app.get("/viewer", include_in_schema=False)
@app.get("/viewer/", include_in_schema=False)
def viewer_page(user: dict = Depends(get_current_user)):
    return FileResponse(viewer_dir / "index.html")


@app.get("/admin", include_in_schema=False)
@app.get("/admin/", include_in_schema=False)
def admin_page(user: dict = Depends(require_roles("owner"))):
    return FileResponse(viewer_dir / "admin.html")


@app.get("/api")
def api_info():
    return {
        "service": "defirmation",
        "version": "1.2.0",
        "authentication": "session-cookie",
        "endpoints": {
            "health": "GET /api/health",
            "auth_status": "GET /api/auth/status",
            "login": "POST /api/auth/login",
            "setup_owner": "POST /api/auth/setup (one-time)",
            "current_user": "GET /api/auth/me",
            "admin_users": "GET/POST /api/admin/users (Owner only)",
            "deform_from_images": "POST /api/deform",
            "deform_from_multi_view": "POST /api/deform/multi-view",
            "deform_from_measurements": "POST /api/deform/measurements",
            "templates": "GET /api/templates",
            "download": "GET /api/output/{filename}",
            "rim_detection": "/api/rim-detection/* (Owner/Editor)",
            "viewer": "GET /viewer/",
        },
    }


@app.get("/api/health")
def health():
    """Liveness probe: process is running and can answer HTTP requests."""
    return {"status": "ok", "version": "1.3.0"}


@app.get("/api/ready")
def readiness():
    """Readiness probe: production templates, model and artifact storage are usable."""
    try:
        candidates = pipeline.matcher._load_candidates()
        names = [template.name for template in candidates]
        if not names:
            raise HTTPException(
                status_code=503,
                detail="No production-ready templates with independent mesh components are available.",
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = OUTPUT_DIR / f".ready-{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        app_env = os.getenv("APP_ENV", "development").strip().lower()
        if app_env in {"production", "prod"} and getattr(pipeline.segmenter, "model", None) is None:
            raise HTTPException(
                status_code=503,
                detail="A trained eyewear segmentation model is required in production.",
            )
        return {
            "status": "ready",
            "version": "1.3.0",
            "production_ready_templates": names,
            "templates_available": len(names),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="Production dependencies are not ready.") from exc


@app.get("/api/templates")
def list_templates(user: dict = Depends(get_current_user)):
    candidates = pipeline.matcher._load_candidates()
    available = [template.name for template in candidates]
    active = _pick_active_template(available)
    return {
        "templates": available,
        "preferred": [t for t in PREFERRED_TEMPLATES if t in available],
        "active": active,
    }


@app.post("/api/deform", deprecated=True)
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image (legacy, uncalibrated path)"),
    side: UploadFile | None = File(None, description="Side product image"),
    top: UploadFile | None = File(None, description="Top product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Legacy image-only path. Production deployments reject uncalibrated scale."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    allow_uncalibrated = os.getenv("ALLOW_UNCALIBRATED_GENERATION", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if app_env in {"production", "prod"} and not allow_uncalibrated:
        raise HTTPException(
            status_code=422,
            detail="Image-only generation cannot establish millimetre scale. Use 4–5 product images with physical measurements.",
        )

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    succeeded = False

    try:
        front_path = work_dir / "front.jpg"
        await _save_validated_image(front, front_path)

        side_path = None
        if side and side.filename:
            side_path = work_dir / "side.jpg"
            await _save_validated_image(side, side_path)

        top_path = None
        if top and top.filename:
            top_path = work_dir / "top.jpg"
            await _save_validated_image(top, top_path)

        result = await run_in_threadpool(
            pipeline.run_from_images,
            front_path,
            side_path,
            out_path,
            color=color,
            template_override=template,
            top_path=top_path,
        )
        _record_output_owner(Path(result["metadata_json"]), user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["created_by"] = user["username"]
        succeeded = True
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image deformation failed; job_id=%s", job_id)
        raise HTTPException(status_code=500, detail=f"Generation failed. Reference job ID: {job_id}.") from exc
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
        if not succeeded:
            _remove_failed_output(out_path)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="Exactly 4–5 product images from multiple view angles"),
    frame_width: float = Form(...),
    lens_width: float = Form(...),
    lens_height: float = Form(...),
    bridge_width: float = Form(...),
    temple_length: float = Form(...),
    rim_thickness: float = Form(1.2),
    temple_curve_angle: float = Form(28.0),
    color: str = Form("#d9a7a2"),
    shape: str | None = Form(None),
    material: str | None = Form(None),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Production multi-view reconstruction with user-supplied physical dimensions."""
    if not 4 <= len(images) <= 5:
        raise HTTPException(status_code=400, detail="Upload exactly 4 or 5 product images.")

    try:
        shape_override = FrameShape(shape) if shape else None
        material_override = FrameMaterial(material) if material else None
        manual_measurements = Measurements(
            frame_width=frame_width,
            lens_width=lens_width,
            lens_height=lens_height,
            bridge_width=bridge_width,
            temple_length=temple_length,
            rim_thickness=rim_thickness,
            temple_curve_angle=temple_curve_angle,
            color=color,
            scale_source="user_supplied",
            scale_calibrated=True,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid physical measurements: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_multi_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    succeeded = False

    try:
        image_paths: list[Path] = []
        seen_hashes: set[str] = set()
        total_bytes = 0
        allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for index, upload in enumerate(images):
            if not upload.filename:
                raise HTTPException(status_code=400, detail=f"Image {index + 1} is missing a filename.")
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in allowed_suffixes:
                suffix = ".jpg"
            image_path = work_dir / f"image_{index}{suffix}"
            file_size = await _save_validated_image(upload, image_path)
            total_bytes += file_size
            if total_bytes > MAX_MULTI_VIEW_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Combined multi-view uploads exceed the total size limit.")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise HTTPException(status_code=400, detail=f"Image {index + 1} duplicates another uploaded image.")
            seen_hashes.add(digest)
            image_paths.append(image_path)

        result = await run_in_threadpool(
            pipeline.run_from_multiple_images,
            image_paths,
            out_path,
            color,
            template,
            manual_measurements,
            shape_override,
            material_override,
        )
        _record_output_owner(Path(result["metadata_json"]), user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["images_used"] = len(image_paths)
        result["created_by"] = user["username"]
        succeeded = True
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Multi-view deformation failed; job_id=%s", job_id)
        raise HTTPException(status_code=500, detail=f"Generation failed. Reference job ID: {job_id}.") from exc
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
        if not succeeded:
            _remove_failed_output(out_path)


@app.post("/api/deform/measurements")
async def deform_from_measurements(
    frame_width: float = Form(...),
    lens_width: float = Form(...),
    lens_height: float = Form(...),
    bridge_width: float = Form(...),
    temple_length: float = Form(...),
    rim_thickness: float = Form(1.2),
    material: str = Form("metal"),
    shape: str = Form("rectangle"),
    nose_pads: bool = Form(True),
    temple_curve_angle: float = Form(28),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Deform a template from known, user-supplied physical measurements."""
    try:
        measurements = Measurements(
            frame_width=frame_width,
            lens_width=lens_width,
            lens_height=lens_height,
            bridge_width=bridge_width,
            temple_length=temple_length,
            rim_thickness=rim_thickness,
            material=FrameMaterial(material),
            shape=FrameShape(shape),
            nose_pads=nose_pads,
            temple_curve_angle=temple_curve_angle,
            color=color,
            scale_source="user_supplied",
            scale_calibrated=True,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid physical measurements: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    succeeded = False
    try:
        result = await run_in_threadpool(
            pipeline.run_from_measurements,
            measurements,
            out_path,
            template,
        )
        _record_output_owner(Path(result["metadata_json"]), user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["created_by"] = user["username"]
        succeeded = True
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Measurement-based deformation failed; job_id=%s", job_id)
        raise HTTPException(status_code=500, detail=f"Generation failed. Reference job ID: {job_id}.") from exc
    finally:
        if not succeeded:
            _remove_failed_output(out_path)


@app.get("/api/output/{filename}")
def download_output(
    filename: str,
    user: dict = Depends(require_roles("owner", "editor", "operator", "viewer")),
):
    """Download a generated GLB only to its owner or the system owner."""
    if not re.fullmatch(r"[0-9a-f]{12}\.glb", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if user["role"] != "owner":
        metadata_path = path.with_suffix(".metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise HTTPException(status_code=404, detail="File not found")
        if metadata.get("created_by_user_id") != user["id"]:
            # Do not reveal whether another user's job ID exists.
            raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type="model/gltf-binary", filename=filename)
