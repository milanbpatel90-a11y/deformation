"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import uuid
import logging
import os
import re
from pydantic import ValidationError
from backend.api.safety import read_image, run_job

logger = logging.getLogger(__name__)
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness
from backend.api import rim_detection_routes
from backend.measurement.suggestions import suggest_measurements
from backend.template_library.compatibility import MeasurementCompatibilityError, measurement_ranges

app = FastAPI(
    title="Defirmation API",
    description="Template deformation pipeline for eyewear virtual try-on",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("DEFIRM_CORS_ORIGINS", "").split(",") if origin.strip()],
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


def parse_measurements(payload: str, color: str) -> Measurements:
    try:
        result = Measurements.model_validate_json(payload)
        result.color = color
        return result
    except ValidationError as exc:
        raise HTTPException(422, "Provide valid manual measurements, positive finite dimensions and hex colours") from exc


def validate_template(name: str | None) -> None:
    if name is not None and name not in library.list_templates():
        raise HTTPException(400, "Unknown or unavailable template")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    catalog = template_readiness(library)
    return JSONResponse({"ready": catalog["ready"], "templates": catalog["templates"],
                         "unavailable": list(catalog["unavailable"])},
                        status_code=200 if catalog["ready"] else 503)


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
    catalog = template_readiness(library)
    available = catalog["templates"]
    active = "rectangle_plastic" if "rectangle_plastic" in available else (available[0] if available else None)
    return {
        "templates": available,
        "planned": library.list_templates(),
        "unavailable": list(catalog["unavailable"]),
        "active": active,
        "ranges": {name: measurement_ranges(library.load(name)) for name in available},
    }


@app.post("/api/measurements/suggest")
async def suggest_from_images(
    images: list[UploadFile] = File(...),
    side: UploadFile | None = File(None),
    top: UploadFile | None = File(None),
):
    if not 1 <= len(images) <= 6:
        raise HTTPException(400, "Upload between 1 and 6 images")
    decoded = [(await read_image(file))[1] for file in images]
    side_image = (await read_image(side))[1] if side else None
    top_image = (await read_image(top))[1] if top else None
    try:
        return await run_job(suggest_measurements, pipeline, decoded, side_image, top_image)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, "Could not estimate supported sizes. Use a clear front photo and check that a template is available.") from exc
    except Exception as exc:
        logger.exception("Measurement suggestion failed")
        raise HTTPException(500, "Size suggestion failed. Retry or enter measurements manually.") from exc


@app.post("/api/deform")
async def deform_from_images(
    front: UploadFile = File(..., description="Front product image"),
    side: UploadFile | None = File(None, description="Side product image"),
    top: UploadFile | None = File(None, description="Top product image"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    measurements: str = Form(..., description="Manual Measurements JSON in millimetres"),
    automatic_appearance: bool = Form(True),
):
    """Upload up to 3 images (front, side, top), use manual measurements, deform template, export GLB."""
    job_id = uuid.uuid4().hex
    # Use OUTPUT_DIR (project-local, ASCII-safe path) instead of system temp
    # to avoid Windows 8.3 tilde paths (PETPOO~1) that break cv2.imread.
    work_dir = OUTPUT_DIR / f"_upload_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        front_path = work_dir / "front.jpg"
        manual = parse_measurements(measurements, color)
        validate_template(template)
        front_bytes, _ = await read_image(front)
        if not front_bytes:
            raise ValueError("Front image upload is empty — please re-select the file and try again.")
        front_path.write_bytes(front_bytes)

        side_path = None
        if side and side.filename:
            side_bytes, _ = await read_image(side)
            if side_bytes:
                side_path = work_dir / "side.jpg"
                side_path.write_bytes(side_bytes)

        top_path = None
        if top and top.filename:
            top_bytes, _ = await read_image(top)
            if top_bytes:
                top_path = work_dir / "top.jpg"
                top_path.write_bytes(top_bytes)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = await run_job(pipeline.run_from_images,
            front_path,
            side_path,
            out_path,
            color=color,
            template_override=template,
            top_path=top_path,
            manual_measurements=manual,
            automatic_appearance=automatic_appearance,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        logger.info("Rejected deformation job %s: %s", job_id, exc)
        raise HTTPException(400, "Input or template is unsuitable for deformation") from exc
    except Exception as exc:
        logger.exception("Deformation job %s failed", job_id)
        raise HTTPException(500, "Deformation failed. Check server logs with job ID " + job_id) from exc
    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/multi-view")
async def deform_from_multiple_images(
    images: list[UploadFile] = File(..., description="List of 4-6 product images from multiple view angles"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    measurements: str = Form(..., description="Manual Measurements JSON in millimetres"),
    automatic_appearance: bool = Form(True),
):
    """Upload multiple product images, classify views, use manual measurements, deform template, and export GLB."""
    job_id = uuid.uuid4().hex
    work_dir = OUTPUT_DIR / f"_upload_multi_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        manual = parse_measurements(measurements, color)
        validate_template(template)
        if not 1 <= len(images) <= 6:
            raise HTTPException(400, "Upload between 1 and 6 images")
        image_paths = []
        for i, file in enumerate(images):
            if not file.filename:
                continue
            file_bytes, _ = await read_image(file)
            if not file_bytes:
                continue
            suffix = ".img"
            file_path = work_dir / f"image_{i}{suffix}"
            file_path.write_bytes(file_bytes)
            image_paths.append(file_path)

        if not image_paths:
            raise ValueError("No non-empty images uploaded")

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = await run_job(pipeline.run_from_multiple_images,
            image_paths,
            out_path,
            color=color,
            template_override=template,
            manual_measurements=manual,
            automatic_appearance=automatic_appearance,
        )
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        logger.info("Rejected deformation job %s: %s", job_id, exc)
        raise HTTPException(400, "Input or template is unsuitable for deformation") from exc
    except Exception as exc:
        logger.exception("Deformation job %s failed", job_id)
        raise HTTPException(500, "Deformation failed. Check server logs with job ID " + job_id) from exc
    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/measurements")
async def deform_from_measurements(
    frame_width: float = Form(...),
    lens_width: float = Form(...),
    lens_height: float = Form(...),
    bridge_width: float = Form(...),
    temple_length: float = Form(...),
    rim_thickness: float = Form(1.2),
    material: str = Form("metal"),
    shape: str = Form("geometric"),
    nose_pads: bool = Form(True),
    temple_curve_angle: float = Form(28),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
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
        raise HTTPException(422, "Invalid manual measurements, material, shape or colour") from exc

    job_id = uuid.uuid4().hex
    out_path = OUTPUT_DIR / f"{job_id}.glb"

    try:
        validate_template(template)
        result = await run_job(pipeline.run_from_measurements, measurements, out_path, template)
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        logger.info("Rejected deformation job %s: %s", job_id, exc)
        raise HTTPException(400, "Input or template is unsuitable for deformation") from exc
    except Exception as exc:
        logger.exception("Deformation job %s failed", job_id)
        raise HTTPException(500, "Deformation failed. Check server logs with job ID " + job_id) from exc


@app.get("/api/output/{filename}")
def download_output(filename: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.(?:glb|metadata\.json)", filename):
        raise HTTPException(404, "File not found")
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "application/json" if filename.endswith(".json") else "model/gltf-binary"
    return FileResponse(path, media_type=media_type, filename=filename)
