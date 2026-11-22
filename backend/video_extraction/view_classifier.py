"""Conservative geometric view heuristics; unknown is preferred over false certainty."""
import numpy as np

def classify_view(bbox, image_shape, masks=None):
    if not bbox: return {"view":"unknown","confidence":0.0,"method":"heuristic"}
    x1,y1,x2,y2=[float(v) for v in bbox]; h,w=image_shape[:2]; bw=max(1,x2-x1); bh=max(1,y2-y1); aspect=bw/bh
    coverage=(bw*bh)/(w*h)
    if coverage<0.006: return {"view":"unknown","confidence":0.1,"method":"heuristic"}
    # A single 2D bbox cannot robustly infer yaw/roll. Front-like classification is the only moderately safe inference.
    if aspect>=2.05:
        confidence=float(np.clip((aspect-1.75)/1.4,0.35,0.72))
        return {"view":"front","confidence":round(confidence,3),"method":"bbox_aspect_heuristic"}
    return {"view":"unknown","confidence":0.2,"method":"insufficient_pose_evidence"}
