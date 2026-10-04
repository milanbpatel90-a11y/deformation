"""Asynchronous production job API backed by Celery/Redis."""
from __future__ import annotations
from pathlib import Path
import uuid
from fastapi import APIRouter,File,HTTPException,UploadFile
from backend.workers.tasks import process_job
from backend.workers.celery_app import celery_app

router=APIRouter(prefix="/api/jobs",tags=["jobs"])
RUNTIME_DIR=Path("/app/runtime") if Path("/app/runtime").exists() else Path("runtime")
UPLOAD_DIR=RUNTIME_DIR/"uploads"; UPLOAD_DIR.mkdir(parents=True,exist_ok=True)

@router.post("")
async def create_job(images:list[UploadFile]=File(default=[]),video:UploadFile|None=File(default=None)):
    if video is None and not images: raise HTTPException(400,"Provide at least one image or a video")
    if video is not None and images: raise HTTPException(400,"Provide images or video, not both")
    job_id=uuid.uuid4().hex
    job_dir=UPLOAD_DIR/job_id; job_dir.mkdir(parents=True)
    inputs=[]; is_video=video is not None
    if video is not None:
        suffix=Path(video.filename or "input.mp4").suffix or ".mp4"; p=job_dir/f"input{suffix}"; p.write_bytes(await video.read()); inputs=[str(p)]
    else:
        if not 1<=len(images)<=8: raise HTTPException(400,"Upload between 1 and 8 images")
        for i,f in enumerate(images):
            suffix=Path(f.filename or "image.jpg").suffix or ".jpg"; p=job_dir/f"image_{i}{suffix}"; data=await f.read()
            if not data: raise HTTPException(400,f"Empty upload: {f.filename}")
            p.write_bytes(data); inputs.append(str(p))
    process_job.apply_async(args=[job_id,inputs,is_video],task_id=job_id)
    return {"job_id":job_id,"status":"queued"}

@router.get("/{job_id}")
def job_status(job_id:str):
    result=celery_app.AsyncResult(job_id); info=result.info if isinstance(result.info,dict) else {}
    if result.state=="PENDING": return {"job_id":job_id,"status":"queued","progress":0.0}
    if result.state=="FAILURE": return {"job_id":job_id,"status":"failed","progress":0.0,"error":str(result.info)}
    if result.state=="SUCCESS": return {"job_id":job_id,"status":"done",**result.result}
    return {"job_id":job_id,"status":"processing","progress":float(info.get("progress",0.0)),"stage":info.get("stage")}
