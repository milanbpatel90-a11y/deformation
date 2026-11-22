"""Run bounded video-to-reference extraction and create reviewable assets."""
from __future__ import annotations
import json,time
from pathlib import Path
import cv2
from .frame_extractor import AdaptiveFrameExtractor
from .frame_ranker import rank_frames
from .measurement_preprocessor import extract_pixel_measurements
from .schemas import VIEW_NAMES
class VideoExtractionPipeline:
 def __init__(self,segmenter,limits=None):
  from .schemas import VideoLimits
  self.limits=limits or VideoLimits.from_env(); self.extractor=AdaptiveFrameExtractor(segmenter,self.limits)
 def run(self,video_path,output_dir,metadata):
  started=time.perf_counter(); root=Path(output_dir); root.mkdir(parents=True,exist_ok=True); candidates,rejected=self.extractor.extract(video_path,root); ranked=rank_frames(candidates,self.limits.max_selected_frames,3); best={}
  for item in ranked: best.setdefault(item["view"],item)
  for view,item in best.items():
   if view!="unknown": self._write_view(item,view,root)
  selected=[v for v in best if v!="unknown"]; missing=[v for v in VIEW_NAMES if v!="unknown" and v not in selected]; warnings=[]
  if "front" not in selected: warnings.append("No confidently classified front view; merchant must manually assign a candidate.")
  if not any(v.startswith("side_") for v in selected): warnings.append("No side reference; temple measurement unavailable.")
  if "top" not in selected: warnings.append("No top reference; frame depth unavailable.")
  report={"status":"awaiting_review","video":metadata,"available_views":selected,"missing_views":missing,"selected_frames":{v:i["frame_id"] for v,i in best.items()},"candidates":[{"frame_id":i["frame_id"],"timestamp_seconds":i["timestamp_seconds"],"view":i["view"],"view_confidence":i["view_confidence"],"quality_score":i["quality_score"],"detection_confidence":i["detection_confidence"],"bbox":i["bbox"],"reasons":i["reasons"],"component_confidences":i["component_confidences"],"image_asset":"candidates/"+i["frame_id"]+".jpg","mask_assets":{p:"candidates/"+i["frame_id"]+"_"+p+".png" for p in i["mask_paths"]}} for i in ranked],"rejected_frames":rejected[:100],"warnings":warnings,"processing_seconds":round(time.perf_counter()-started,3)}
  quality={"candidate_count":len(candidates),"selected_count":len(ranked),"rejected_count":len(rejected),"views":{v:sum(i["view"]==v for i in ranked) for v in VIEW_NAMES}}; pixels=extract_pixel_measurements(best,self._mask_loader)
  for name,payload in (("extraction_report.json",report),("quality_report.json",quality),("measurements.json",pixels)): (root/name).write_text(json.dumps(payload,indent=2),encoding="utf-8")
  return {"extraction_report":report,"quality_report":quality,"measurements":pixels}
 @staticmethod
 def _mask_loader(candidate,part):
  path=candidate.get("mask_paths",{}).get(part); return cv2.imread(path,cv2.IMREAD_GRAYSCALE) if path else None
 @staticmethod
 def _write_view(item,view,root):
  image=cv2.imread(item["image_path"]); 
  for folder in ("extracted","masks","overlays"): (root/folder).mkdir(parents=True,exist_ok=True)
  cv2.imwrite(str(root/"extracted"/(view+".png")),image); overlay=image.copy(); alpha=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)*0
  for part,path in item["mask_paths"].items():
   mask=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
   if mask is None: continue
   cv2.imwrite(str(root/"masks"/(view+"_"+part+".png")),mask); alpha=cv2.bitwise_or(alpha,mask); tint=image.copy()*0; tint[mask>0]=(60,180,255); overlay=cv2.addWeighted(overlay,.8,tint,.2,0)
  cv2.imwrite(str(root/"overlays"/(view+"_overlay.png")),overlay)
  if alpha.any(): rgba=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA); rgba[:,:,3]=alpha; cv2.imwrite(str(root/"extracted"/(view+"_transparent.png")),rgba)
