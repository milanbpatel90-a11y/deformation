"""Pixel-only geometric measurements and explicit millimetre calibration."""
from __future__ import annotations
import cv2, numpy as np

def _bbox(mask):
    if mask is None or not np.any(mask): return None
    ys,xs=np.where(mask>0); return int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
def _record(value,view,confidence,note=None): return {"value":round(float(value),3) if value is not None else None,"unit":"px","confidence":confidence,"source_view":view,"calibration_status":"uncalibrated","kind":"pixel_measurement","note":note}
def extract_pixel_measurements(selected,mask_loader):
    m={}; warnings=[]; front=selected.get("front")
    if front:
        boxes={p:_bbox(mask_loader(front,p)) for p in ("rim","left_lens","right_lens","bridge")}; rb,lb,rr,bb=boxes["rim"],boxes["left_lens"],boxes["right_lens"],boxes["bridge"]
        if rb: m["frame_width"]=_record(rb[2]-rb[0],"front",.72)
        if lb and rr: m["lens_width"]=_record(float(np.median([lb[2]-lb[0],rr[2]-rr[0]])),"front",.68); m["lens_height"]=_record(float(np.median([lb[3]-lb[1],rr[3]-rr[1]])),"front",.68)
        if bb: m["bridge_width"]=_record(bb[2]-bb[0],"front",.68)
        if rb:
            mask=mask_loader(front,"rim"); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter=sum(cv2.arcLength(c,True) for c in contours); area=float(np.count_nonzero(mask))
                if perimeter>0: m["rim_thickness"]=_record(2*area/perimeter,"front",.3,"rough estimate; merchant verification required")
    else: warnings.append("Front reference missing; front-view measurements unavailable.")
    side=selected.get("side_left") or selected.get("side_right")
    if side:
        box=_bbox(mask_loader(side,"temple"))
        if box: m["temple_length"]=_record(max(box[2]-box[0],box[3]-box[1]),side.get("view","side"),.5,"projected length; view-specific scale required")
    else: warnings.append("Side reference missing; temple length unavailable.")
    for key,view in (("frame_width","front"),("lens_width","front"),("lens_height","front"),("bridge_width","front"),("temple_length","side"),("rim_thickness","front")): m.setdefault(key,_record(None,view,0,"required semantic mask/reference unavailable"))
    return {"measurements":m,"warnings":warnings,"absolute_dimensions_available":False}
def apply_calibration(pixel_report,calibration=None,physical_measurements=None):
    import copy
    report=copy.deepcopy(pixel_report); calibration=calibration or {}; physical_measurements=physical_measurements or {}
    fields={"frame_width":"frame_width_mm","lens_width":"lens_width_mm","lens_height":"lens_height_mm","bridge_width":"bridge_width_mm","temple_length":"temple_length_mm","rim_thickness":"rim_thickness_mm"}
    for key,field in fields.items():
        if physical_measurements.get(field) is not None:
            value=float(physical_measurements[field])
            if value<=0: raise ValueError(field+" must be positive")
            report["measurements"][key]={"value":value,"unit":"mm","confidence":1.0,"source_view":"merchant_supplied","calibration_status":"merchant_supplied","kind":"physical_measurement"}
    for item in report["measurements"].values():
        if item["unit"]!="px" or item["value"] is None: continue
        ref=calibration.get(item["source_view"])
        if ref:
            mm=float(ref.get("reference_length_mm",0)); px=float(ref.get("measured_length_px",0))
            if mm>0 and px>0: item.update(value=round(item["value"]*mm/px,3),unit="mm",confidence=min(item["confidence"],.78),calibration_status="calibrated_from_reference",kind="calibrated_estimate")
    missing=[k for k in fields if report["measurements"][k]["unit"]!="mm" or report["measurements"][k]["value"] is None]
    report["missing_physical_measurements"]=missing; report["absolute_dimensions_available"]=not missing
    if missing: report["warnings"].append("Physical dimensions missing calibration: "+", ".join(missing))
    return report
def to_deformation_measurements(report):
    missing=report.get("missing_physical_measurements",[])
    if missing: raise ValueError("Millimetre calibration is incomplete for: "+", ".join(missing))
    return {k:float(report["measurements"][k]["value"]) for k in ("frame_width","lens_width","lens_height","bridge_width","temple_length","rim_thickness")}
