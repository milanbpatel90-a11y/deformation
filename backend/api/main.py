"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.api import rim_detection_routes

logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = int(os.environ.get("DEFIRM_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "DEFIRM_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Defirmation API",
    description="Template deformation pipeline for eyewear virtual try-on",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Content-Type"],
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = DeformationPipeline()
library = TemplateLibrary()

viewer_dir = Path(__file__).resolve().parents[2] / "viewer"
if viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")

app.include_router(rim_detection_routes.router)


@app.get("/")
def root():
    return {
        "service": "defirmation",
        "version": "1.0.0",
        "endpoints": {
            "deform_from_images": "POST /api/deform",
            "deform_from_multi_view": "POST /api/deform/multi-view",
            "deform_from_measurements": "POST /api/deform/measurements",
            "templates": "GET /api/templates",
            "download": "GET /api/output/{filename}",
            "detect_rim": "POST /api/rim-detection/detect",
            "apply_rim_to_template": "POST /api/rim-detection/apply-to-template",
            "rim_status": "GET /api/rim-detection/status",
        },
    }


@app.get("/api/templates")
def list_templates():
    available = library.list_templates()
    active = "rectangle_plastic" if "rectangle_plastic" in available else (available[0] if available else None)
    return {
        "templates": available,
        "planned": available,
        "active": active,
    }


@app.post("/api/deform")
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image"),
    side: UploadFile | None = File(None, description="Side product image"),
    top: UploadFile | None = File(None, description="Top product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
):
    """Upload up to 3 images (front, side, top), detect measurements, deform template, export GLB."""
    job_id = uuid.uuid4().hex[:12]
    # Use OUTPUT_DIR (project-local, ASCII-safe path) instead of system temp
    # to avoid Windows 8.3 tilde paths (PETPOO~1) that break cv2.imread.
    work_dir = OUTPUT_DIR / f"_upload_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        front_path = work_dir / "front.jpg"
        front_bytes = await front.read()
        if not front_bytes:
            raise ValueError("Front image upload is empty — please re-select the file and try again.")
        if len(front_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Front image exceeds upload size limit")
        front_path.write_bytes(front_bytes)

        side_path = None
        if side and side.filename:
            side_bytes = await side.read()
            if side_bytes:
                if len(side_bytes) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Side image exceeds upload size limit")
                side_path = work_dir / "side.jpg"
                side_path.write_bytes(side_bytes)

        top_path = None
        if top and top.filename:
            top_bytes = await top.read()
            if top_bytes:
                if len(top_bytes) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Top image exceeds upload size limit")
                top_path = work_dir / "top.jpg"
                top_path.write_bytes(top_bytes)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = await run_in_threadpool(
            pipeline.run_from_images,
            front_path,
            side_path,
            out_path,
            color,
            template,
            top_path,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Single-view deformation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="Exactly 4-5 product images from multiple view angles"),
    frame_width: float = Form(...),
    lens_width: float = Form(...),
    lens_height: float = Form(...),
    bridge_width: float = Form(...),
    temple_length: float = Form(...),
    rim_thickness: float = Form(1.2),
    temple_curve_angle: float = Form(28.0),
    color: str = Form("#000000"),
    shape: str | None = Form(None),
    material: str | None = Form(None),
    template: str | None = Form(None),
):
    """Production multi-view deformation using user-entered physical measurements."""
    if not 4 <= len(images) <= 5:
        raise HTTPException(status_code=400, detail="Upload exactly 4 or 5 images")

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
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid reconstruction input: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_multi_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_paths: list[Path] = []
        seen_hashes: set[str] = set()

        for i, file in enumerate(images):
            if not file.filename:
                raise HTTPException(status_code=400, detail=f"Image {i + 1} is missing a filename")
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail=f"Image {i + 1} is empty")
            if len(file_bytes) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail=f"{file.filename} exceeds upload size limit")

            digest = hashlib.sha256(file_bytes).hexdigest()
            if digest in seen_hashes:
                raise HTTPException(status_code=400, detail=f"{file.filename} duplicates another uploaded image")
            seen_hashes.add(digest)

            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".jpg"
            file_path = work_dir / f"image_{i}{suffix}"
            file_path.write_bytes(file_bytes)
            image_paths.append(file_path)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
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
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multi-view deformation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    shape: str = Form("geometric"),
    nose_pads: bool = Form(True),
    temple_curve_angle: float = Form(28),
    color: str = Form("#d9a7a2"),
    template: str = Form("rectangle_plastic"),
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
        result = await run_in_threadpool(pipeline.run_from_measurements, measurements, out_path, template)
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/output/{filename}")
def download_output(filename: str):
    if not filename.endswith(".glb") or any(ch not in "0123456789abcdef.glb" for ch in filename.lower()):
        raise HTTPException(status_code=400, detail="Invalid output filename")
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)
