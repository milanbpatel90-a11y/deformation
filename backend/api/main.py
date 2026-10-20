"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.video_pipeline import VideoTo3DPipeline
from backend.api import rim_detection_routes

app = FastAPI(
    title="Defirmation API",
    description="Template deformation pipeline for eyewear virtual try-on",
    version="1.1.0",
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
video_pipeline = VideoTo3DPipeline(pipeline=pipeline)
library = TemplateLibrary()

viewer_dir = Path(__file__).resolve().parents[2] / "viewer"
if viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")

app.include_router(rim_detection_routes.router)


@app.get("/")
def root():
    return {
        "service": "defirmation",
        "version": "1.1.0",
        "endpoints": {
            "deform_from_images": "POST /api/deform",
            "deform_from_multi_view": "POST /api/deform/multi-view",
            "deform_from_video": "POST /api/deform/video",
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
    return {"templates": available, "planned": available, "active": active}


@app.post("/api/deform")
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image"),
    side: UploadFile | None = File(None, description="Side product image"),
    top: UploadFile | None = File(None, description="Top product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
):
    """Upload product images, deform a template, and export GLB."""
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
        import traceback
        raise HTTPException(status_code=500, detail=f"{exc}\n\nTraceback:\n{traceback.format_exc()}") from exc
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="List of 4-6 product images from multiple view angles"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
):
    """Upload multiple product images, fuse measurements, deform template, and export GLB."""
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
            file_path = work_dir / f"image_{i}{suffix}"
            file_path.write_bytes(file_bytes)
            image_paths.append(file_path)

        if not image_paths:
            raise ValueError("No non-empty images uploaded")

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = pipeline.run_from_multiple_images(
            image_paths,
            out_path,
            color=color,
            template_override=template,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        import traceback
        raise HTTPException(status_code=500, detail=f"{exc}\n\nTraceback:\n{traceback.format_exc()}") from exc
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/video")
async def deform_from_video(
    video: UploadFile = File(..., description="Controlled 360-degree eyewear video"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    reference_width_mm: float | None = Form(None),
):
    """Upload a glasses video and automatically create a template-deformed GLB.

    The video is sampled into multiple views. The pipeline segments each usable
    view, selects strong front-facing evidence, robustly fuses measurements,
    deforms the best matching template, and exports a GLB plus video metadata.
    """
    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / f"_upload_video_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not video.filename:
            raise ValueError("No video file supplied.")

        suffix = Path(video.filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
            raise ValueError("Unsupported video format. Use MP4, MOV, M4V, AVI, or WebM.")

        video_bytes = await video.read()
        if not video_bytes:
            raise ValueError("Video upload is empty — please select the video again.")

        video_path = work_dir / f"input{suffix}"
        video_path.write_bytes(video_bytes)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = video_pipeline.process(
            video_path=video_path,
            output_path=out_path,
            color=color,
            template_override=template,
            reference_width_mm=reference_width_mm,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except Exception as exc:
        import traceback
        raise HTTPException(status_code=500, detail=f"{exc}\n\nTraceback:\n{traceback.format_exc()}") from exc
    finally:
        import shutil
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
