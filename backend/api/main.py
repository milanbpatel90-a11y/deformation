"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import logging
import os
import re
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

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
        raise HTTPException(status_code=413, detail="An image exceeds the 20 MiB upload limit.")
    try:
        with Image.open(BytesIO(contents)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the processing limit.")
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="An uploaded file is not a valid supported image.") from exc
    destination.write_bytes(contents)
    return len(contents)


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


@app.get("/api/ready")
@app.get("/api/health")
def health():
    """Liveness + basic readiness for production monitoring."""
    available = library.list_templates()
    active = _pick_active_template(available)
    return {
        "status": "ok",
        "version": "1.2.0",
        "templates_available": len(available),
        "active_template": active,
        "preferred_templates": [t for t in PREFERRED_TEMPLATES if t in available],
    }


@app.get("/api/templates")
def list_templates(user: dict = Depends(get_current_user)):
    available = library.list_templates()
    active = _pick_active_template(available)
    return {
        "templates": available,
        "preferred": [t for t in PREFERRED_TEMPLATES if t in available],
        "active": active,
    }


@app.post("/api/deform")
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image"),
    side: UploadFile | None = File(None, description="Side product image"),
    top: UploadFile | None = File(None, description="Top product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Upload front/side/top images, validate them, deform a template, and export a GLB."""
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

        result = pipeline.run_from_images(
            front_path,
            side_path,
            out_path,
            color=color,
            template_override=template,
            top_path=top_path,
        )
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
    images: list[UploadFile] = File(..., description="4–5 product images from multiple view angles"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Fuse 4–5 validated product images, deform the best-fit template, and export a GLB."""
    if not 4 <= len(images) <= 5:
        raise HTTPException(status_code=400, detail="Multi-view generation requires 4–5 product images.")

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_multi_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    succeeded = False

    try:
        image_paths: list[Path] = []
        total_bytes = 0
        allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for index, upload in enumerate(images):
            if not upload.filename:
                raise HTTPException(status_code=400, detail="Every multi-view upload must include an image file.")
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in allowed_suffixes:
                suffix = ".jpg"
            image_path = work_dir / f"image_{index}{suffix}"
            total_bytes += await _save_validated_image(upload, image_path)
            if total_bytes > MAX_MULTI_VIEW_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Combined multi-view uploads exceed the total size limit.")
            image_paths.append(image_path)

        result = pipeline.run_from_multiple_images(
            image_paths,
            out_path,
            color=color,
            template_override=template,
        )
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
    frame_width: float = Form(145),
    lens_width: float = Form(54),
    lens_height: float = Form(50),
    bridge_width: float = Form(18),
    temple_length: float = Form(140),
    rim_thickness: float = Form(1.2),
    material: str = Form("metal"),
    shape: str = Form("rectangle"),
    nose_pads: bool = Form(True),
    temple_curve_angle: float = Form(28),
    color: str = Form("#d9a7a2"),
    template: str = Form("rectangle_metal"),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Deform a template from known physical measurements (no images required)."""
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    succeeded = False
    try:
        result = pipeline.run_from_measurements(measurements, out_path, template)
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
    """Download only API-generated GLB artifacts addressed by opaque job IDs."""
    if not re.fullmatch(r"[0-9a-f]{12}\.glb", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)


@app.get("/api/output/{filename}")
def download_output(
    filename: str,
    user: dict = Depends(require_roles("owner", "editor", "operator", "viewer")),
):
    # Prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=safe_name)
