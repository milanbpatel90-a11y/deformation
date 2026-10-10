"""FastAPI endpoints for template deformation VTO pipeline."""

from __future__ import annotations

import uuid
import logging
import os
import re
import json
from pydantic import ValidationError
from backend.api.safety import read_image, run_job, save_video_upload, video_suffix

logger = logging.getLogger(__name__)
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.pipeline.video_pipeline import VideoDeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness
from backend.api import rim_detection_routes
from backend.measurement.suggestions import suggest_measurements
from backend.template_library.compatibility import (
    MeasurementCompatibilityError, measurement_ranges, dependent_ranges, validate_combination)

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

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output"
PRODUCTION_MODE = os.getenv("DEFIRM_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
OUTPUT_DIR = OUTPUT_ROOT / ("production" if PRODUCTION_MODE else "development")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = DeformationPipeline()
# Share the loaded YOLO weights, template library and matcher rather than
# loading a second copy of each.
video_pipeline = VideoDeformationPipeline.sharing(pipeline)
library = TemplateLibrary()

viewer_dir = Path(__file__).resolve().parents[2] / "viewer"
if viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")

app.include_router(rim_detection_routes.router)


@app.exception_handler(MeasurementCompatibilityError)
async def incompatible_measurements(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc), "ranges": exc.ranges})


def input_ranges(measurements, template=None):
    name = template or "GT_001"
    base = measurement_ranges(library.load(name)) if name in library.list_templates() else {}
    return dependent_ranges(measurements, base)


def check_combination(measurements, template=None):
    validate_combination(measurements, input_ranges(measurements, template))


@app.post("/api/measurements/ranges")
def calculate_ranges(measurements: Measurements, template: str | None = None):
    validate_template(template)
    return {"ranges": input_ranges(measurements, template),
            "constraint": "frame_width >= 2*lens_width + bridge_width + 2*rim_thickness"}


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
                         "unavailable": list(catalog["unavailable"]),
                         "production_mode": PRODUCTION_MODE},
                        status_code=200 if catalog["ready"] else 503)


@app.get("/")
def root():
    return {
        "service": "defirmation",
        "version": "1.0.0",
        "endpoints": {
            "deform_from_images": "POST /api/deform",
            "deform_from_multi_view": "POST /api/deform/multi-view",
            "deform_from_orbit_video": "POST /api/deform/video",
            "deform_from_measurements": "POST /api/deform/measurements",
            "templates": "GET /api/templates",
            "download": "GET /api/output/{filename}",
            "download_orbit_frame": "GET /api/output/frames/{job_id}/{filename}",
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
    labels = {}
    for name in available:
        descriptor = library.bundle_dir / name / "metadata/template.json"
        if descriptor.is_file():
            try:
                labels[name] = json.loads(descriptor.read_text(encoding="utf-8")).get(
                    "display_name", "Gold Template (GT_001)" if name == "GT_001" else name)
            except (OSError, json.JSONDecodeError):
                labels[name] = name
        else:
            labels[name] = "Gold Template (GT_001)" if name == "GT_001" else name
    return {
        "templates": available,
        "labels": labels,
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


@app.post("/api/measurements/suggest/video")
async def suggest_from_orbit_video(
    video: UploadFile = File(..., description="360-degree orbit video to measure"),
    min_views: int = Form(5),
    target_views: int = Form(8),
    max_views: int = Form(10),
    target_fps: float = Form(6.0),
    reference_width_mm: float | None = Form(None),
):
    """Measure an orbit video *without* deforming it.

    Runs the same decode -> gate -> select -> segment -> measure -> fuse half of
    the video pipeline that the deform endpoint uses, so the viewer can show the
    auto-measured sizes and the template that would be chosen before the user
    commits to a deformation. No GLB is produced and nothing is exported.
    """
    job_id = uuid.uuid4().hex
    work_dir = OUTPUT_DIR / f"_upload_suggest_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not 1 <= min_views <= target_views <= max_views <= 48:
            raise HTTPException(
                400, "View counts must satisfy 1 <= min_views <= target_views <= max_views <= 48"
            )
        if reference_width_mm is not None and not 20.0 <= float(reference_width_mm) <= 250.0:
            raise HTTPException(400, "reference_width_mm must be between 20 and 250 mm")

        suffix = video_suffix(video.filename)
        video_path = work_dir / f"orbit{suffix}"
        await save_video_upload(video, video_path)

        estimate = await run_job(
            video_pipeline.estimate_from_video,
            video_path,
            min_views=min_views,
            max_views=max_views,
            target_views=target_views,
            target_fps=target_fps,
            reference_width_mm=reference_width_mm,
        )
        measurements = estimate["measurements"]
        # Score the template here so the viewer can show it, using the same
        # matcher the deformation path will use -- not a second selection rule.
        features = video_pipeline.feature_extractor.from_measurements(
            measurements, estimate["style"]
        )
        match = video_pipeline.matcher.match(features, None, measurements=measurements)
        template_name = match.best.template.name
        confidence = estimate["confidence"]
        scale_source = str(estimate["scale"]["source"]).replace("_", " ")
        return {
            "job_id": job_id,
            "measurements": measurements.model_dump(),
            "template": template_name,
            "ranges": input_ranges(measurements, template_name),
            "confidence": confidence,
            "scale": estimate["scale"],
            "selection": estimate["selection_summary"],
            "view_distribution": estimate["view_distribution"],
            "rear_available": video_pipeline.view_classifier.REAR_AVAILABLE,
            "model": {
                "type": video_pipeline.segmenter.model_type,
                "classes": video_pipeline.segmenter.class_count,
                "guard": estimate["model_guard"],
            },
            "timings": video_pipeline._timings(estimate["report"], None),
            "pipeline": estimate["report"].to_dict(),
            "note": (
                f"Measured from {estimate['selection_summary']['selected_count']} orbit views "
                f"({confidence['level']} confidence). Sizes are anchored to the {scale_source} "
                f"of {estimate['scale']['reference_width_mm']} mm unless you supply a reference width."
            ),
            "adjustments": list(confidence.get("notes", [])),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        logger.info("Rejected orbit estimate job %s: %s", job_id, exc)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Orbit estimate job %s failed", job_id)
        raise HTTPException(500, "Video measurement failed. Check server logs with job ID " + job_id) from exc
    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)


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
        check_combination(manual, template)
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
        result["ranges"] = input_ranges(Measurements(**result["measurements"]), result.get("template"))
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["manifest_url"] = f"/api/output/{job_id}.manifest.json"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError:
        raise
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
        check_combination(manual, template)
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
        result["ranges"] = input_ranges(Measurements(**result["measurements"]), result.get("template"))
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["manifest_url"] = f"/api/output/{job_id}.manifest.json"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError:
        raise
    except ValueError as exc:
        logger.info("Rejected deformation job %s: %s", job_id, exc)
        raise HTTPException(400, "Input or template is unsuitable for deformation") from exc
    except Exception as exc:
        logger.exception("Deformation job %s failed", job_id)
        raise HTTPException(500, "Deformation failed. Check server logs with job ID " + job_id) from exc
    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/deform/video")
async def deform_from_orbit_video(
    video: UploadFile = File(..., description="8-12 second 360-degree orbit video of the eyewear"),
    color: str = Form("#d9a7a2"),
    template: str | None = Form(None),
    measurements: str | None = Form(None, description="Optional manual Measurements JSON to override the fused sizes"),
    automatic_appearance: bool = Form(True),
    min_views: int = Form(5, description="Fewest views to accept for fusion"),
    target_views: int = Form(8, description="Preferred number of views to fuse"),
    max_views: int = Form(10, description="Most views to fuse"),
    target_fps: float = Form(6.0, description="Frame sampling rate for the decode"),
    reference_width_mm: float | None = Form(
        None,
        description="Known real frame width in mm; scales the estimate off the extractor default",
    ),
):
    """Deform a template from a 360-degree orbit video.

    Decode -> quality gate -> bounded candidate pool -> 1-class eyewear
    segmentation + visibility gate -> coverage-aware selection of 5-10 views ->
    per-view measurement -> weighted median + MAD fusion -> the existing
    DeformationPipeline -> validated GLB + manifest.

    Absolute scale is not measured from the video: ``frame_width`` is anchored to
    the extractor's reference width unless ``reference_width_mm`` is supplied.
    """
    job_id = uuid.uuid4().hex
    # Project-local, ASCII-safe paths: Windows 8.3 tilde paths break OpenCV's
    # path-based file APIs.
    work_dir = OUTPUT_DIR / f"_upload_video_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = OUTPUT_DIR / f"{job_id}_frames"

    try:
        validate_template(template)
        if not 1 <= min_views <= target_views <= max_views <= 48:
            raise HTTPException(
                400, "View counts must satisfy 1 <= min_views <= target_views <= max_views <= 48"
            )
        if reference_width_mm is not None and not 20.0 <= float(reference_width_mm) <= 250.0:
            raise HTTPException(400, "reference_width_mm must be between 20 and 250 mm")
        manual = None
        if measurements:
            manual = parse_measurements(measurements, color)
            check_combination(manual, template)

        suffix = video_suffix(video.filename)
        video_path = work_dir / f"orbit{suffix}"
        await save_video_upload(video, video_path)

        out_path = OUTPUT_DIR / f"{job_id}.glb"
        result = await run_job(
            video_pipeline.run_from_video,
            video_path,
            out_path,
            color=color,
            template_override=template,
            manual_measurements=manual,
            automatic_appearance=automatic_appearance,
            min_views=min_views,
            max_views=max_views,
            target_views=target_views,
            target_fps=target_fps,
            preview_dir=preview_dir,
            reference_width_mm=reference_width_mm,
        )
        result["ranges"] = input_ranges(Measurements(**result["measurements"]), result.get("template"))
        result["job_id"] = job_id
        result["selected_views"] = (result.get("video") or {}).get("selection", {}).get("selected_count")
        result["confidence"] = (result.get("video") or {}).get("confidence", {})
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["manifest_url"] = f"/api/output/{job_id}.manifest.json"
        result["video_manifest_url"] = f"/api/output/{job_id}.video.json"
        attach_preview_urls(result, job_id)
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError:
        raise
    except ValueError as exc:
        # These messages are written for the person holding the camera, so they
        # are surfaced verbatim rather than collapsed into a generic failure.
        logger.info("Rejected orbit video job %s: %s", job_id, exc)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Orbit video job %s failed", job_id)
        raise HTTPException(500, "Video deformation failed. Check server logs with job ID " + job_id) from exc
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
        # Manual-size submissions without a selected template use Gold, which
        # remains the stable default even as new catalog templates are added.
        selected_template = template or ("GT_001" if "GT_001" in library.list_templates() else None)
        check_combination(measurements, selected_template)
        result = await run_job(pipeline.run_from_measurements, measurements, out_path, selected_template)
        result["ranges"] = input_ranges(Measurements(**result["measurements"]), result.get("template"))
        result["job_id"] = job_id
        result["download_url"] = f"/api/output/{job_id}.glb"
        result["manifest_url"] = f"/api/output/{job_id}.manifest.json"
        return result
    except HTTPException:
        raise
    except MeasurementCompatibilityError:
        raise
    except ValueError as exc:
        logger.info("Rejected deformation job %s: %s", job_id, exc)
        raise HTTPException(400, "Input or template is unsuitable for deformation") from exc
    except Exception as exc:
        logger.exception("Deformation job %s failed", job_id)
        raise HTTPException(500, "Deformation failed. Check server logs with job ID " + job_id) from exc


def attach_preview_urls(result: dict, job_id: str) -> None:
    """Replace absolute preview paths with bearer URLs the viewer can fetch."""
    previews = (result.get("video") or {}).get("previews")
    if not previews:
        return
    # Output URLs are bearer links, so the server directory is not disclosed.
    previews.pop("directory", None)
    base = f"/api/output/frames/{job_id}"
    names = previews.get("frames") or []
    previews["frame_urls"] = [f"{base}/{name}" for name in names]
    previews["grid_url"] = f"{base}/{previews['grid']}" if previews.get("grid") else None
    if previews["grid_url"]:
        # Convenience alias for the documented response contract.
        result["contact_sheet_url"] = previews["grid_url"]


@app.get("/api/output/{filename}")
def download_output(filename: str):
    # .video.json is the orbit manifest; it is served by the same confined route
    # as the GLB and the deformation manifest.
    if not re.fullmatch(
        r"[A-Za-z0-9_-]+\.(?:glb|metadata\.json|manifest\.json|video\.json)", filename
    ):
        raise HTTPException(404, "File not found")
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "application/json" if filename.endswith(".json") else "model/gltf-binary"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/output/frames/{job_id}/{filename}")
def download_preview_frame(job_id: str, filename: str):
    """Serve one selected orbit frame, or the contact sheet, for the viewer."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise HTTPException(404, "File not found")
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", filename):
        raise HTTPException(404, "File not found")
    directory = (OUTPUT_DIR / f"{job_id}_frames").resolve()
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="image/jpeg")
