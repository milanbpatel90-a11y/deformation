"""Adapter around the existing segmenter; semantic component names are preserved."""
import cv2

class GlassesDetector:
    def __init__(self, segmenter): self.segmenter=segmenter
    def detect(self, image):
        if self.segmenter is None: return {"detected":False,"confidence":0.0,"bbox":None,"masks":{},"component_confidences":{},"model_type":"unavailable"}
        detailed=self.segmenter.segment_detailed(image) if hasattr(self.segmenter,"segment_detailed") else None
        if detailed and detailed.get("semantic_model_available"):
            detections=detailed.get("detections",[]); masks=detailed.get("masks",{})
            if not detections: return {"detected":False,"confidence":0.0,"bbox":None,"masks":masks,"component_confidences":{},"model_type":detailed.get("model_type")}
            boxes=[d["bbox"] for d in detections]; bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            conf={name:max((d["confidence"] for d in detections if d["class_name"]==name),default=0.0) for name in ("rim","temple","bridge","left_lens","right_lens","nose_pad")}
            return {"detected":True,"confidence":max(d["confidence"] for d in detections),"bbox":bbox,"masks":masks,"component_confidences":conf,"model_type":detailed.get("model_type")}
        # Legacy merged-mask fallback can establish an object region but cannot claim semantic parts.
        legacy=self.segmenter.segment(image); mask=legacy.get("full")
        if mask is None or not mask.any(): return {"detected":False,"confidence":0.0,"bbox":None,"masks":{},"component_confidences":{},"model_type":legacy.get("model_type","unknown")}
        x,y,w,h=cv2.boundingRect(mask); return {"detected":True,"confidence":0.25,"bbox":[x,y,x+w,y+h],"masks":{},"component_confidences":{},"model_type":legacy.get("model_type","legacy_nonsemantic")}
