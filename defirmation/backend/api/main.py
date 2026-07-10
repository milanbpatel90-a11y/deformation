"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

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
    description="Template deformation pipeline for eyewear virtual try-on",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
            "deform_from_measurements": "POST /api/deform/measurements",
            "templates": "GET /api/templates",
            "download": "GET /api/output/{filename}",
        },
    }


@app.get("/api/templates")
def list_templates():
    available = library.list_templates()
    return {
        "templates": available,
        "planned": list(library.TEMPLATE_REGISTRY.keys()),
        "active": "geometric_metal",
    }


@app.post("/api/deform")
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image"),
    side: UploadFile | None = File(None, description="Side product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
):
    """Upload 1-3 images, detect measurements, deform template, export GLB."""
    job_id = uuid.uuid4().hex[:12]
    work_dir = Path(tempfile.mkdtemp(prefix=f"defirm_{job_id}_"))

    try:
        front_path = work_dir / "front.jpg"
        with open(front_path, "wb") as f:
            shutil.copyfileobj(front.file, f)

        side_path = None
        if side and side.filename:
            side_path = work_dir / "side.jpg"
            with open(side_path, "wb") as f:
                shutil.copyfileobj(side.file, f)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = pipeline.run_from_images(
            front_path,
            side_path,
            out_path,
            color=color,
            template_override=template,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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
    template: str = Form("geometric_metal"),
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
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/output/{filename}")
def download_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)
