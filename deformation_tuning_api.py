"""
FastAPI server for lens contour deformation tuning.
Runs on port 8001 (main VTO API uses 8000).
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

import cv2
import numpy as np

from backend.fusion.view_classifier import ViewClassifier
from backend.pipeline import DeformationPipeline
from eyewear_vto_toolkit import DeformationCalibrationWorkflow, TemplateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DEFAULT_YOLO = PROJECT_ROOT / "runs" / "segment" / "eyewear_seg" / "weights" / "best.pt"

app = FastAPI(title="Defirmation Deformation Tuning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TuningState:
    def __init__(self, template_manager: TemplateManager, calibration: DeformationCalibrationWorkflow):
        self.template_manager = template_manager
        self.calibration = calibration
        self.current_template: str | None = None
        self.current_image: str | None = None
        self.detected_rim_vertices: list[int] = []
        self.edge_detection_confidence = 0.0
        self.contour_fit_error = 0.0
        self.proposed_rim_pull = 0.65
        self.calibration_history: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_template": self.current_template,
            "detected_rim_vertices_count": len(self.detected_rim_vertices),
            "edge_detection_confidence": self.edge_detection_confidence,
            "contour_fit_error": self.contour_fit_error,
            "proposed_rim_pull": self.proposed_rim_pull,
            "calibration_history_length": len(self.calibration_history),
        }


state: TuningState | None = None
_reconstruction_pipeline: DeformationPipeline | None = None


def _get_reconstruction_pipeline() -> DeformationPipeline:
    global _reconstruction_pipeline
    if _reconstruction_pipeline is None:
        _reconstruction_pipeline = DeformationPipeline()
    return _reconstruction_pipeline


def _overlay_png_data_uri(image: np.ndarray, mask: np.ndarray) -> str:
    tint = np.zeros_like(image)
    tint[mask > 0] = (0, 200, 110)
    overlay = cv2.addWeighted(image, 0.72, tint, 0.28, 0)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise ValueError("Failed to encode segmentation overlay")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _agreement(per_view: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    fields = {
        "frame_width": 3.0,
        "lens_width": 2.0,
        "lens_height": 2.0,
        "bridge_width": 2.0,
        "temple_length": 5.0,
    }
    agreement: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for field, threshold in fields.items():
        if field == "temple_length":
            relevant = [v for v in per_view if v["view_type"] == "side"] or per_view
        else:
            relevant = [v for v in per_view if v["view_type"] in {"front", "perspective"}] or per_view

        values = [
            float(v["measurements"][field])
            for v in relevant
            if v["measurements"].get(field) is not None
        ]
        sample_count = len(values)
        if sample_count < 2:
            agreement[field] = {
                "std_dev": None,
                "flag": None,
                "sample_count": sample_count,
                "status": "insufficient_samples",
            }
            warnings.append(
                f"{field}: insufficient relevant views for agreement check "
                f"({sample_count} sample; need at least 2)"
            )
            continue

        std_dev = round(float(np.std(values)), 2)
        flagged = std_dev > threshold
        agreement[field] = {
            "std_dev": std_dev,
            "flag": flagged,
            "sample_count": sample_count,
            "status": "disagreement" if flagged else "agreement",
        }
        if flagged:
            warnings.append(
                f"{field} varies ±{std_dev}mm across relevant views (threshold {threshold}mm)"
            )

    return agreement, warnings


def _init_state() -> TuningState:
    yolo_path = os.environ.get("YOLO_MODEL_PATH", str(DEFAULT_YOLO))
    tm = TemplateManager(TEMPLATES_DIR)
    cal = DeformationCalibrationWorkflow(tm, yolo_path if Path(yolo_path).exists() else None)
    return TuningState(tm, cal)


@app.on_event("startup")
async def startup() -> None:
    global state
    state = _init_state()
    logger.info("Tuning API ready — templates dir: %s", TEMPLATES_DIR)


@app.get("/api/health")
async def health():
    return {"status": "ok", "state": state.to_dict() if state else None}


@app.get("/api/templates")
async def list_templates():
    if not state:
        raise HTTPException(status_code=500, detail="State not initialized")
    templates = []
    for name, metadata in state.template_manager.templates.items():
        glb_exists = state.template_manager.resolved_glb_path(name)
        templates.append(
            {
                "name": name,
                "style": metadata.style,
                "bridge_width_mm": metadata.bridge_width_mm,
                "rim_pull_strength": metadata.rim_pull_strength,
                "glb_exists": glb_exists.exists(),
                "last_calibrated": metadata.last_calibrated,
            }
        )
    return {"templates": templates}


@app.post("/api/templates/{template_name}/select")
async def select_template(template_name: str):
    if not state:
        raise HTTPException(status_code=500)
    if template_name not in state.template_manager.templates:
        raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
    state.current_template = template_name
    meta = state.template_manager.templates[template_name]
    state.proposed_rim_pull = meta.rim_pull_strength
    return {
        "selected": template_name,
        "style": meta.style,
        "bridge_width_mm": meta.bridge_width_mm,
        "temple_length_mm": meta.temple_length_mm,
        "ffd_grid_dims": meta.ffd_grid_dims,
        "rim_pull_strength": meta.rim_pull_strength,
    }


@app.post("/api/calibration/upload-image")
async def upload_calibration_image(file: UploadFile = File(...)):
    if not state or not state.current_template:
        raise HTTPException(status_code=400, detail="Select template first")
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        state.current_image = tmp.name
    return {"filename": file.filename, "size": len(contents), "stored_path": state.current_image}


@app.post("/api/calibration/detect-rim")
async def detect_rim_contour():
    if not state or not state.current_image or not state.current_template:
        raise HTTPException(status_code=400, detail="Upload image and select template first")
    try:
        result = state.calibration.detect_rim_from_image(state.current_image)
        state.detected_rim_vertices = result["detected_vertices"]
        state.edge_detection_confidence = result["edge_detection_confidence"]
        state.contour_fit_error = result["contour_fit_error"]
        state.proposed_rim_pull = result["proposed_rim_pull_strength"]

        state.calibration.log_calibration_attempt(
            state.current_template,
            state.current_image,
            state.detected_rim_vertices,
            {
                "edge_confidence": state.edge_detection_confidence,
                "fit_error": state.contour_fit_error,
            },
        )
        return result
    except Exception as exc:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/calibration/adjust-pull-strength")
async def adjust_pull_strength(strength: float = Query(...)):
    if not state:
        raise HTTPException(status_code=400)
    state.proposed_rim_pull = max(0.0, min(1.0, strength))
    return {"rim_pull_strength": state.proposed_rim_pull}


@app.post("/api/calibration/preview-deformation")
async def preview_deformation():
    if not state or not state.current_template:
        raise HTTPException(status_code=400)
    return {
        "template": state.current_template,
        "rim_pull_strength": state.proposed_rim_pull,
        "affected_vertices": state.detected_rim_vertices,
        "deformation_magnitude": round(state.proposed_rim_pull * 2.5, 2),
        "preview_ready": bool(state.detected_rim_vertices),
    }


@app.post("/api/calibration/confirm-and-save")
async def confirm_calibration():
    if not state or not state.current_template:
        raise HTTPException(status_code=400)
    state.template_manager.update_calibration(
        state.current_template,
        state.detected_rim_vertices,
        state.proposed_rim_pull,
    )
    state.calibration_history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "template": state.current_template,
            "rim_pull_strength": state.proposed_rim_pull,
            "vertices_count": len(state.detected_rim_vertices),
            "edge_confidence": state.edge_detection_confidence,
        }
    )
    return {
        "success": True,
        "template": state.current_template,
        "saved_rim_pull_strength": state.proposed_rim_pull,
    }


@app.get("/api/calibration/history")
async def get_calibration_history():
    if not state:
        raise HTTPException(status_code=500)
    return {"history": state.calibration_history}


@app.post("/api/calibration/batch-templates")
async def batch_create_templates():
    if not state:
        raise HTTPException(status_code=500)
    count = state.template_manager.create_batch_stubs()
    return {"success": True, "templates_created": count, "templates": list(state.template_manager.templates.keys())}



@app.post("/api/reconstruct/multi-view")
async def reconstruct_multi_view(
    images: list[UploadFile] = File(..., description="4-5 images of one pair of glasses"),
    template_name: str | None = Form(None),
):
    """Dashboard-compatible multi-view reconstruction endpoint."""
    if not 4 <= len(images) <= 5:
        raise HTTPException(status_code=400, detail="Upload exactly 4 or 5 images")

    pipeline = _get_reconstruction_pipeline()
    view_classifier = ViewClassifier()
    job_id = uuid.uuid4().hex[:12]

    try:
        with tempfile.TemporaryDirectory(prefix=f"defirm_multiview_{job_id}_") as tmp_dir:
            work_dir = Path(tmp_dir)
            image_paths: list[Path] = []
            original_names: list[str] = []

            for index, upload in enumerate(images):
                raw = await upload.read()
                if not raw:
                    raise HTTPException(status_code=400, detail=f"Image {index + 1} is empty")
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                    suffix = ".jpg"
                path = work_dir / f"view_{index}{suffix}"
                path.write_bytes(raw)
                image_paths.append(path)
                original_names.append(upload.filename or path.name)

            per_view: list[dict[str, Any]] = []
            warnings: list[str] = []

            for filename, path in zip(original_names, image_paths):
                image = pipeline._imread(path)
                if image is None:
                    raise HTTPException(status_code=400, detail=f"Cannot decode {filename}")

                masks = pipeline.segmenter.segment(image)
                mask = masks["front"]
                view_type = view_classifier.classify_view(image, mask)
                style = pipeline.classifier.classify_style(image, mask)
                measurements, _ = pipeline.measurer.extract_from_images(
                    image,
                    side=None,
                    mask=mask,
                    shape=style.shape,
                    material=style.material,
                    nose_pads=style.nose_pads,
                    color="#d9a7a2",
                )

                confidence = float(masks.get("confidence", 0.0))
                model_type = str(masks.get("model_type", "unknown"))
                if model_type == "opencv_fallback":
                    warnings.append(f"{filename}: YOLO unavailable; OpenCV fallback used")
                elif confidence < 0.75:
                    warnings.append(f"{filename}: low mask confidence ({confidence:.2f})")

                per_view.append(
                    {
                        "filename": filename,
                        "view_type": view_type,
                        "overlay_png_base64": _overlay_png_data_uri(image, mask),
                        "mask_confidence": round(confidence, 4),
                        "measurements": {
                            "frame_width": measurements.frame_width,
                            "lens_width": measurements.lens_width,
                            "lens_height": measurements.lens_height,
                            "bridge_width": measurements.bridge_width,
                            "temple_length": measurements.temple_length,
                        },
                    }
                )

            glb_path = work_dir / f"{job_id}.glb"
            result = await run_in_threadpool(
                pipeline.run_from_multiple_images,
                image_paths,
                glb_path,
                "#d9a7a2",
                template_name,
            )

            agreement, agreement_warnings = _agreement(per_view)
            warnings.extend(agreement_warnings)

            fused = result["measurements"]
            consolidated = {
                key: fused[key]
                for key in ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length")
                if key in fused
            }

            model_uri = (
                "data:model/gltf-binary;base64,"
                + base64.b64encode(glb_path.read_bytes()).decode("ascii")
            )

            return {
                "job_id": job_id,
                "views": per_view,
                "consolidated_measurements": consolidated,
                "measurement_agreement": agreement,
                "warnings": warnings,
                "model_glb_base64": model_uri,
                "template": result.get("template"),
                "pipeline": result.get("pipeline", []),
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multi-view reconstruction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/state")
async def get_state():
    if not state:
        raise HTTPException(status_code=500)
    return state.to_dict()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = PROJECT_ROOT / "toolkit" / "tuning_ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "<h1>Defirmation Tuning API</h1><p>UI file missing at toolkit/tuning_ui.html</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
