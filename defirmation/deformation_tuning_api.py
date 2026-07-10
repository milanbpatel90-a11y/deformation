"""
FastAPI server for lens contour deformation tuning.
Runs on port 8001 (main VTO API uses 8000).
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
