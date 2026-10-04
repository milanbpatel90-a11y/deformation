"""Image quality checks for production multi-view ingestion."""
from __future__ import annotations
import cv2
import numpy as np

def laplacian_variance(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0: return 0.0
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def glare_ratio(frame: np.ndarray, threshold: int=245) -> float:
    if frame is None or frame.size == 0: return 1.0
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((gray >= threshold).mean())

def frame_score(frame: np.ndarray) -> dict[str,float]:
    blur=laplacian_variance(frame); glare=glare_ratio(frame)
    sharp=float(np.clip(blur/300.0,0.0,1.0))
    clean=float(np.clip(1.0-glare/0.08,0.0,1.0))
    return {"blur":blur,"glare":glare,"score":round(0.65*sharp+0.35*clean,4)}

def is_acceptable(frame: np.ndarray,min_score: float=0.45)->bool:
    return frame_score(frame)["score"]>=min_score
