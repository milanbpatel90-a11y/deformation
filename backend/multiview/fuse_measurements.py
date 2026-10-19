"""Robust fusion utilities for per-view numeric measurements."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Measurement:
    frame_width: float; lens_width: float; lens_height: float; bridge_width: float
    temple_length: float; temple_curve_angle: float=28.0; confidence: float=1.0; view_angle: float=0.0

def _weighted_median(values: np.ndarray,weights: np.ndarray)->float:
    order=np.argsort(values); v=values[order]; w=np.maximum(weights[order],1e-9)
    return float(v[np.searchsorted(np.cumsum(w),np.sum(w)/2.0)])

def robust_estimate(values: Sequence[float],weights: Sequence[float])->tuple[float,float]:
    v=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
    med=_weighted_median(v,w); mad=float(np.median(np.abs(v-med)))
    scale=max(1.4826*mad,1e-6); inlier=np.abs(v-med)<=3.0*scale
    if inlier.sum()<2: inlier=np.ones(len(v),dtype=bool)
    est=_weighted_median(v[inlier],w[inlier])
    residual=float(np.average(np.abs(v[inlier]-est),weights=w[inlier]))
    conf=float(np.clip(1.0-residual/max(abs(est),1e-6),0.0,1.0))
    return est,conf

def fuse_measurements(measurements: Sequence[Measurement])->dict:
    if not measurements: raise ValueError("No measurements to fuse")
    out={}
    for key in ("frame_width","lens_width","lens_height","bridge_width","temple_length","temple_curve_angle"):
        vals=[getattr(m,key) for m in measurements]
        weights=[max(m.confidence,0.0)*(max(np.cos(np.deg2rad(m.view_angle)),0.05) if key!="temple_length" else max(np.sin(np.deg2rad(abs(m.view_angle))),0.25)) for m in measurements]
        est,conf=robust_estimate(vals,weights); out[key]=round(est,3); out[f"{key}_confidence"]=round(conf,3)
    out["overall_confidence"]=round(float(np.mean([out[f"{k}_confidence"] for k in ("frame_width","lens_width","lens_height","bridge_width","temple_length","temple_curve_angle")])),3)
    out["num_views"]=len(measurements)
    return out

def measurements_to_dict(ms: Sequence[Measurement])->list[dict]: return [asdict(m) for m in ms]
