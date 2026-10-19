"""Estimate a coarse yaw class from an eyewear segmentation mask."""
from __future__ import annotations
import cv2
import numpy as np

def classify_angle(frame: np.ndarray, mask: np.ndarray|None=None)->float:
    if mask is None:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); _,mask=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    ys,xs=np.where(mask>0)
    if len(xs)<50:return 0.0
    x0,x1=float(xs.min()),float(xs.max()); y0,y1=float(ys.min()),float(ys.max())
    width=max(x1-x0,1.0); height=max(y1-y0,1.0)
    aspect=width/height
    cx=float(xs.mean()); left=float((xs<cx).sum()); right=float((xs>=cx).sum())
    balance=min(left,right)/max(left,right,1.0)
    if aspect>=2.0 and balance>=0.80:return 0.0
    yaw=float(np.clip((1.0-balance)*90.0,0.0,90.0))
    return yaw if left>right else -yaw
