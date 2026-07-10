"""
FastAPI Integration Module for Rim Detection
Endpoints to integrate rim detection into main eyewear-vto backend.
Add these to backend.api.main or create backend.api.rim_detection_routes.py
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
import json

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/rim-detection", tags=["rim-detection"])


# ============================================================================
# Rim Detection Engine (Stateless)
# ============================================================================

class RimDetectionEngine:
    """Rim detection using OpenCV (fallback) or YOLOv8-Seg"""
    
    def __init__(self, yolo_model_path: Optional[str] = None):
        """
        Initialize rim detection engine.
        
        Args:
            yolo_model_path: Path to trained YOLOv8 model (optional)
                           If None, uses OpenCV threshold fallback
        """
        self.yolo_model = None
        self.use_yolo = False
        
        if yolo_model_path and Path(yolo_model_path).exists():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_model_path)
                self.use_yolo = True
                logger.info(f"✓ Loaded YOLOv8 model from {yolo_model_path}")
            except ImportError:
                logger.warning("YOLOv8 not available, using OpenCV fallback")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}, using fallback")
    
    def get_glasses_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get binary segmentation mask of glasses from image.
        
        Uses:
        1. YOLOv8-Seg if model loaded
        2. OpenCV threshold as fallback
        
        Args:
            image: BGR image (numpy array, H×W×3)
        
        Returns:
            Binary mask (H×W, 0=background, 255=glasses)
        """
        
        if self.use_yolo and self.yolo_model:
            return self._get_mask_yolov8(image)
        else:
            return self._get_mask_opencv(image)
    
    def _get_mask_yolov8(self, image: np.ndarray) -> np.ndarray:
        """YOLOv8-Seg inference"""
        try:
            results = self.yolo_model(image, conf=0.5, imgsz=640)
            
            if results and len(results) > 0 and results[0].masks is not None:
                # Get segmentation mask
                mask = results[0].masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                
                # Resize to match image size
                h, w = image.shape[:2]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                
                return mask
        except Exception as e:
            logger.warning(f"YOLOv8 inference failed: {e}, using fallback")
        
        return self._get_mask_opencv(image)
    
    def _get_mask_opencv(self, image: np.ndarray) -> np.ndarray:
        """OpenCV threshold fallback"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold (works for synthetic images)
        # For real images, use edge detection or color-based segmentation
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_rim_contour(
        self,
        mask: np.ndarray,
        n_vertices: int = 8,
    ) -> Dict:
        """
        Detect rim contour from mask.
        
        Returns:
            {
                "vertices": [[x1, y1], [x2, y2], ...],  (8 points)
                "edge_confidence": 0.0-1.0,
                "fit_error": 0.0-1.0,
                "contour_area": int,
                "success": bool,
            }
        """
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return {
                "vertices": [],
                "edge_confidence": 0.0,
                "fit_error": 1.0,
                "contour_area": 0,
                "success": False,
                "error": "No contours found",
            }
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 500:
            return {
                "vertices": [],
                "edge_confidence": 0.0,
                "fit_error": 1.0,
                "contour_area": int(area),
                "success": False,
                "error": f"Contour area too small: {area}",
            }
        
        # Fit polygon
        perimeter = cv2.arcLength(largest, True)
        epsilon = 0.02 * perimeter
        polygon = cv2.approxPolyDP(largest, epsilon, True)
        
        # Resample to n_vertices
        vertices = self._resample_polygon(polygon.reshape(-1, 2), n_vertices)
        
        # Measure metrics
        edge_conf = self._measure_edge_sharpness(mask, largest)
        fit_err = self._measure_polygon_fit_error(largest, polygon)
        
        return {
            "vertices": vertices.tolist(),
            "edge_confidence": float(edge_conf),
            "fit_error": float(fit_err),
            "contour_area": int(area),
            "success": True,
        }
    
    def _resample_polygon(self, points: np.ndarray, n: int) -> np.ndarray:
        """Resample polygon to exactly n points"""
        
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]
        
        new_distances = np.linspace(0, total_length, n)
        
        resampled = np.array([
            np.interp(new_distances, cumulative, points[:, 0]),
            np.interp(new_distances, cumulative, points[:, 1]),
        ]).T
        
        return resampled.astype(np.int32)
    
    def _measure_edge_sharpness(self, mask: np.ndarray, contour: np.ndarray) -> float:
        """Edge detection confidence"""
        
        edges = cv2.Canny(mask, 50, 150)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, 2)
        
        overlap = cv2.bitwise_and(edges, contour_mask)
        overlap_ratio = np.count_nonzero(overlap) / (np.count_nonzero(contour_mask) + 1e-6)
        
        return min(1.0, overlap_ratio * 1.5)
    
    def _measure_polygon_fit_error(self, contour: np.ndarray, polygon: np.ndarray) -> float:
        """Polygon fit error"""
        
        error = cv2.matchShapes(contour, polygon, cv2.CONTOURS_MATCH_I3, 0.0)
        normalized = min(1.0, error / 10.0)
        
        return normalized
    
    def estimate_rim_pull_strength(
        self,
        edge_confidence: float,
        fit_error: float,
    ) -> float:
        """Estimate rim pull strength from metrics"""
        
        strength = edge_confidence * (1.0 - fit_error)
        strength = np.clip(strength, 0.1, 0.9)
        
        return float(strength)


# Global engine instance
_rim_engine: Optional[RimDetectionEngine] = None


def get_rim_engine() -> RimDetectionEngine:
    """Get or initialize rim detection engine"""
    
    global _rim_engine
    
    if _rim_engine is None:
        # Search for trained YOLOv8 model in all possible locations
        candidate_paths = [
            Path("runs/segment/eyewear_seg/weights/best.pt"),
            Path("runs/segment/train/weights/best.pt"),
            Path("runs/segment/runs/segment/eyewear_seg/weights/best.pt"),
            Path("runs/detect/eyewear_seg/weights/best.pt"),
        ]
        
        yolo_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                yolo_path = candidate
                logger.info(f"Found YOLO weights at: {yolo_path}")
                break
        
        if yolo_path is None:
            logger.warning("No trained YOLO weights found, using OpenCV fallback")
        
        _rim_engine = RimDetectionEngine(
            yolo_model_path=str(yolo_path) if yolo_path else None
        )
    
    return _rim_engine


# ============================================================================
# FastAPI Endpoints
# ============================================================================

@router.post("/detect")
async def detect_rim_from_image(
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Detect rim contour from uploaded image.
    
    Returns:
        {
            "success": bool,
            "vertices": [[x, y], ...],  (8 points in image coordinates)
            "edge_confidence": 0.0-1.0,
            "fit_error": 0.0-1.0,
            "rim_pull_strength": 0.1-0.9,
            "image_size": [width, height],
        }
    """
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        h, w = image.shape[:2]
        
        # Get rim detection engine
        engine = get_rim_engine()
        
        # Get segmentation mask
        mask = engine.get_glasses_mask(image)
        
        # Detect rim contour
        result = engine.detect_rim_contour(mask, n_vertices=8)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Detection failed"))
        
        # Estimate rim pull strength
        rim_pull = engine.estimate_rim_pull_strength(
            result["edge_confidence"],
            result["fit_error"],
        )
        
        return JSONResponse({
            "success": True,
            "vertices": result["vertices"],
            "edge_confidence": result["edge_confidence"],
            "fit_error": result["fit_error"],
            "rim_pull_strength": rim_pull,
            "image_size": [w, h],
        })
    
    except Exception as e:
        logger.error(f"Rim detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-to-template")
async def apply_rim_detection_to_template(
    file: UploadFile = File(...),
    template_name: str = "geometric_metal",
) -> JSONResponse:
    """
    Detect rim, estimate pull strength, and suggest template calibration update.
    
    Returns:
        {
            "template_name": str,
            "detected_rim_pull_strength": 0.1-0.9,
            "recommendation": "Update template with this value",
            "detection_metrics": {...},
        }
    """
    
    try:
        # Run rim detection first
        contents = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Reopen as UploadFile-like
        from fastapi import File as FastAPIFile
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.COLOR_BGR2BGR)
        
        engine = get_rim_engine()
        mask = engine.get_glasses_mask(image)
        result = engine.detect_rim_contour(mask)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail="Rim detection failed")
        
        rim_pull = engine.estimate_rim_pull_strength(
            result["edge_confidence"],
            result["fit_error"],
        )
        
        return JSONResponse({
            "template_name": template_name,
            "detected_rim_pull_strength": rim_pull,
            "recommendation": f"Update {template_name} with rim_pull_strength={rim_pull:.2f}",
            "detection_metrics": {
                "edge_confidence": result["edge_confidence"],
                "fit_error": result["fit_error"],
            },
        })
    
    except Exception as e:
        logger.error(f"Template calibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_rim_detection_status() -> JSONResponse:
    """Check status of rim detection engine"""
    
    engine = get_rim_engine()
    
    return JSONResponse({
        "engine_ready": True,
        "using_yolo": engine.use_yolo,
        "yolo_model_path": "runs/detect/eyewear_seg/weights/best.pt",
        "yolo_available": engine.yolo_model is not None,
        "fallback": "OpenCV threshold" if not engine.use_yolo else None,
    })
