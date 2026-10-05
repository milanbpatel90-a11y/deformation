"""Select sharp, temporally distributed keyframes from an orbit video."""
from __future__ import annotations
from pathlib import Path
import cv2
from .quality_gate import frame_score,is_acceptable

def extract_keyframes(video_path: str|Path,out_dir: str|Path,target:int=5,min_gap_frames:int=15)->list[str]:
    vp=Path(video_path); od=Path(out_dir); od.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(vp))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {vp}")
    candidates=[]; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%3==0 and is_acceptable(frame): candidates.append((idx,frame,frame_score(frame)["score"]))
        idx+=1
    cap.release()
    if not candidates: raise ValueError("No acceptable video frames found")
    # Maximize quality while enforcing temporal separation.
    candidates.sort(key=lambda x:x[2],reverse=True)
    picked=[]
    for item in candidates:
        if all(abs(item[0]-p[0])>=min_gap_frames for p in picked): picked.append(item)
        if len(picked)>=target: break
    picked.sort(key=lambda x:x[0])
    paths=[]
    for i,(frame_idx,frame,score) in enumerate(picked):
        p=od/f"frame_{i:02d}_idx{frame_idx}.jpg"
        if not cv2.imwrite(str(p),frame,[cv2.IMWRITE_JPEG_QUALITY,95]): raise IOError(f"Failed to write {p}")
        paths.append(str(p))
    return paths
