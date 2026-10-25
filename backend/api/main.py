"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.api import rim_detection_routes

app = FastAPI(
    title="Defirmation API",
    description="Template deformation pipeline for eyewear virtual try-on — product images → 3D GLB",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Prefer topology-cleaned production templates (see templates/FIXED_TEMPLATES_README.txt)
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
if viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")

app.include_router(rim_detection_routes.router)


def _pick_active_template(available: list[str]) -> str | None:
    for name in PREFERRED_TEMPLATES:
        if name in available:
            return name
    return available[0] if available else None


@app.get("/")
def root():
    return {
        "service": "defirmation",
        "version": "1.1.0",
        "endpoints": {
            "health": "GET /api/health",
            "deform_from_images": "POST /api/deform",
            "deform_from_multi_view": "POST /api/deform/multi-view",
            "deform_from_measurements": "POST /api/deform/measurements",
            "templates": "GET /api/templates",
            "download": "GET /api/output/{filename}",
            "detect_rim": "POST /api/rim-detection/detect",
            "apply_rim_to_template": "POST /api/rim-detection/apply-to-template",
            "rim_status": "GET /api/rim-detection/status",
            "viewer": "GET /viewer/",
        },
    }


@app.get("/api/health")
def health():
    """Liveness + basic readiness for production monitoring."""
    available = library.list_templates()
    active = _pick_active_template(available)
    model_type = getattr(pipeline.segmenter, "model_type", "unknown")
    ready = bool(active) and model_type != "opencv_fallback"
    return {
        "status": "ok" if ready else "degraded",
        "version": "1.1.0",
        "ready": ready,
        "templates_available": len(available),
        "active_template": active,
        "preferred_templates": [t for t in PREFERRED_TEMPLATES if t in available],
        "segmentation_model": model_type,
        "calibration_note": "Provide frame_width_mm from product specifications for calibrated physical dimensions.",
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/api/templates")
def list_templates():
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
    frame_width_mm: float | None = Form(None, ge=100, le=180),
):
    """Upload front/side/top images and generate a quality-gated GLB.

    Supply frame_width_mm from the manufacturer's product specifications to
    calibrate absolute dimensions instead of using the 135 mm default assumption.
    """
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
            reference_width_mm=frame_width_mm,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Eyewear image-to-GLB pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Pipeline failed. Check server logs for diagnostic details.",
        ) from exc
    finally:
        import shutil as _shutil

        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="List of 2–8 product images from multiple view angles"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
):
    """Upload multiple product images, classify views, fuse measurements, deform template, and export GLB.

    Recommended: 4–5 images (front, left, right, 3/4 angle, optional top).
    """
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
            # Normalize common image types
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
            reference_width_mm=frame_width_mm,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["images_used"] = len(image_paths)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Eyewear image-to-GLB pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Pipeline failed. Check server logs for diagnostic details.",
        ) from exc
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
):
    """Deform template directly from known measurements (no images required).

    Preferred when manufacturer specs are known — most accurate size for try-on.
    """
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
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/output/{filename}")
def download_output(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=safe_name)
