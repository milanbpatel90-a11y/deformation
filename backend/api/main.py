"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from backend.supabase_auth import (
    SESSION_COOKIE_NAME,
    auth_router,
    csrf_token_is_valid,
    get_current_user,
    get_current_user_optional,
    require_roles,
    set_session_cookies,
)
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.api import rim_detection_routes, product_routes


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
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
    )


@app.middleware("http")
async def csrf_protection(request: Request, call_next):
    """Require the double-submit CSRF token on authenticated state changes."""
    safe_methods = {"GET", "HEAD", "OPTIONS"}
    exempt_paths = {"/api/auth/session"}
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
    response = await call_next(request)
    refreshed = getattr(request.state, "supabase_refreshed_session", None)
    if refreshed:
        set_session_cookies(response, refreshed["access_token"], refreshed["refresh_token"])
    return response


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


def _record_model_owner(job_id: str, user: dict) -> None:
    """Persist the authenticated owner beside each generated GLB."""
    import json
    manifest = OUTPUT_DIR / f"{job_id}.ownership.json"
    manifest.write_text(json.dumps({"user_id": user["id"], "email": user["email"]}), encoding="utf-8")


def _assert_model_access(filename: str, user: dict) -> Path:
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename or not safe_name.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Invalid model filename")
    path = OUTPUT_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    if user["role"] != "owner":
        import json
        manifest = OUTPUT_DIR / f"{path.stem}.ownership.json"
        try:
            ownership = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="Model not found")
        if ownership.get("user_id") != user["id"]:
            raise HTTPException(status_code=404, detail="Model not found")
    return path

app.include_router(auth_router)
app.include_router(rim_detection_routes.router)
app.include_router(product_routes.router)


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


@app.get("/dashboard", include_in_schema=False)
def product_dashboard(user: dict = Depends(get_current_user)):
    return FileResponse(viewer_dir / "dashboard.html")


@app.get("/viewer", include_in_schema=False)
@app.get("/viewer/", include_in_schema=False)
def viewer_page(user: dict = Depends(get_current_user)):
    return FileResponse(viewer_dir / "index.html")


@app.get("/api")
def api_info():
    return {
        "service": "defirmation",
        "version": "1.2.0",
        "authentication": "Supabase Google OAuth",
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
            "download": "GET /api/output/{filename} (Owner only)",
            "products": "GET/POST /api/products; PATCH/DELETE /api/products/{id}",
            "product_model": "GET /api/products/{id}/model (owner account or admin)",
            "product_download": "GET /api/products/{id}/download (Owner only)",
            "dashboard": "GET /dashboard",
            "rim_detection": "/api/rim-detection/* (Owner/Editor)",
            "viewer": "GET /viewer/",
        },
    }


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
        "output_dir": str(OUTPUT_DIR),
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
    """Upload up to 3 images (front, side, top), detect measurements, deform template, export GLB."""
    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        front_path = work_dir / "front.jpg"
        front_bytes = await front.read()
        if not front_bytes:
            raise ValueError("Front image upload is empty — please re-select the file and try again.")
        front_path.write_bytes(front_bytes)

        side_path = None
        if side and side.filename:
            side_bytes = await side.read()
            if side_bytes:
                side_path = work_dir / "side.jpg"
                side_path.write_bytes(side_bytes)

        top_path = None
        if top and top.filename:
            top_bytes = await top.read()
            if top_bytes:
                top_path = work_dir / "top.jpg"
                top_path.write_bytes(top_bytes)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = pipeline.run_from_images(
            front_path,
            side_path,
            out_path,
            color=color,
            template_override=template,
            top_path=top_path,
        )
        _record_model_owner(job_id, user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/model/{job_id}.glb"
        result["created_by"] = user["username"]
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n\nTraceback:\n{tb}") from exc
    finally:
        import shutil as _shutil

        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="List of 2–8 product images from multiple view angles"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    user: dict = Depends(require_roles("owner", "editor", "operator")),
):
    """Upload multiple product images, classify views, fuse measurements, deform template, and export GLB."""
    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_multi_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_paths = []
        for i, file in enumerate(images):
            if not file.filename:
                continue
            file_bytes = await file.read()
            if not file_bytes:
                continue
            suffix = Path(file.filename).suffix or ".jpg"
            if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            file_path = work_dir / f"image_{i}{suffix}"
            file_path.write_bytes(file_bytes)
            image_paths.append(file_path)

        if not image_paths:
            raise ValueError("No non-empty images uploaded. Please select 4–5 product photos of the glasses.")
        if len(image_paths) < 2:
            raise ValueError(
                f"Only {len(image_paths)} image received. Multi-view needs at least 2 images "
                "(front + side or angle). For best results upload 4–5 views."
            )
        if len(image_paths) > 8:
            raise ValueError("Maximum 8 images allowed per request.")

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = pipeline.run_from_multiple_images(
            image_paths,
            out_path,
            color=color,
            template_override=template,
        )
        _record_model_owner(job_id, user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/model/{job_id}.glb"
        result["images_used"] = len(image_paths)
        result["created_by"] = user["username"]
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n\nTraceback:\n{tb}") from exc
    finally:
        import shutil as _shutil

        _shutil.rmtree(work_dir, ignore_errors=True)


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
    """Deform template directly from known measurements (no images required)."""
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

    try:
        result = pipeline.run_from_measurements(measurements, out_path, template)
        _record_model_owner(job_id, user)
        result["job_id"] = job_id
        result["download_url"] = f"/api/model/{job_id}.glb"
        result["created_by"] = user["username"]
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/output/{filename}")
def download_output(
    filename: str,
    user: dict = Depends(require_roles("owner")),
):
    path = _assert_model_access(filename, user)
    safe_name = path.name
    return FileResponse(path, media_type="model/gltf-binary", filename=safe_name)


@app.get("/api/model/{filename}")
def stream_model_for_viewer(
    filename: str,
    user: dict = Depends(require_roles("owner", "editor", "operator", "viewer")),
):
    """Serve a model inline for the authenticated in-browser viewer, not as a download."""
    path = _assert_model_access(filename, user)
    return FileResponse(path, media_type="model/gltf-binary", headers={"Content-Disposition": "inline", "Cache-Control": "private, no-store"})
