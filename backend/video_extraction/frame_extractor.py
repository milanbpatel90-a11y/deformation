"""Adaptive time sampling and candidate scoring; never decodes full video into RAM."""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
from .duplicate_filter import perceptual_hash,hash_distance
from .exceptions import VideoProcessingError
from .glasses_detector import GlassesDetector
from .quality_analyzer import analyze_quality
from .schemas import VideoLimits
from .view_classifier import classify_view

class AdaptiveFrameExtractor:
    def __init__(self,segmenter,limits:VideoLimits): self.detector=GlassesDetector(segmenter); self.limits=limits
    def extract(self,video_path,output_dir):
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise VideoProcessingError("Unable to open uploaded video.")
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); duration=float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)/fps if fps>0 else 0
        interval=max(self.limits.sample_interval_seconds,duration/max(self.limits.max_sample_frames,1))
        candidates=[]; rejected=[]; hashes=[]; t=0.0; index=0
        frames_dir=Path(output_dir)/"candidates"; frames_dir.mkdir(parents=True,exist_ok=True)
        while t<=duration and index<self.limits.max_sample_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
            if not ok or frame is None: t+=interval; continue
            h,w=frame.shape[:2]; factor=min(1.0,1280/max(h,w)); work=cv2.resize(frame,None,fx=factor,fy=factor) if factor<1 else frame
            quality=analyze_quality(work); detection=self.detector.detect(work); reasons=list(quality["reasons"])
            bbox=detection["bbox"]
            if not detection["detected"]: reasons.append("glasses_not_detected")
            else:
                x1,y1,x2,y2=bbox; coverage=max(0,x2-x1)*max(0,y2-y1)/(work.shape[0]*work.shape[1])
                if coverage<self.limits.min_bbox_coverage: reasons.append("glasses_too_small")
                if x1<2 or y1<2 or x2>=work.shape[1]-2 or y2>=work.shape[0]-2: reasons.append("glasses_truncated")
            view=classify_view(bbox,work.shape,detection.get("masks"))
            phash=perceptual_hash(work); duplicate=next((i for i,old in enumerate(hashes) if hash_distance(phash,old)<=self.limits.duplicate_hash_distance),None)
            if duplicate is not None: reasons.append("near_duplicate")
            keep= detection["detected"] and not any(r in reasons for r in ("blurry","underexposed","overexposed","glasses_too_small","glasses_truncated","near_duplicate")) and quality["quality_score"]>=self.limits.min_quality_score
            frame_id=f"f{index:04d}"; image_path=frames_dir/f"{frame_id}.jpg"
            if keep:
                cv2.imwrite(str(image_path),work,[cv2.IMWRITE_JPEG_QUALITY,94]); hashes.append(phash)
                mask_paths={}
                for part,mask in detection.get("masks",{}).items():
                    if mask is not None and mask.any():
                        path=frames_dir/f"{frame_id}_{part}.png"; cv2.imwrite(str(path),mask); mask_paths[part]=str(path)
                candidates.append({"frame_id":frame_id,"timestamp_seconds":round(t,3),"image_path":str(image_path),"view":view["view"],"view_confidence":view["confidence"],"quality_score":quality["quality_score"],"quality":quality,"detection_confidence":detection["confidence"],"bbox":bbox,"component_confidences":detection.get("component_confidences",{}),"mask_paths":mask_paths,"reasons":[]})
            else: rejected.append({"frame_id":frame_id,"timestamp_seconds":round(t,3),"reasons":reasons,"quality_score":quality["quality_score"],"detection_confidence":detection["confidence"]})
            t+=interval; index+=1
        cap.release()
        if not candidates: raise VideoProcessingError("No usable glasses frames were found. Record a steadier, closer video with front and side views.")
        return candidates,rejected
