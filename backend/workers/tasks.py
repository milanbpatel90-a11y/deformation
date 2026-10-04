"""Celery task that reuses the repository's real DeformationPipeline."""
from __future__ import annotations
import os,shutil
from pathlib import Path
from .celery_app import celery_app
from backend.pipeline import DeformationPipeline
from backend.video.extract_frames import extract_keyframes

def _upload(path:Path,job_id:str)->str:
    bucket=os.getenv("S3_BUCKET")
    if bucket:
        import boto3
        client=boto3.client("s3",region_name=os.getenv("AWS_REGION"))
        key=f"models/{job_id}/{path.name}"
        client.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":"model/gltf-binary"})
        return os.getenv("S3_PUBLIC_BASE",f"https://{bucket}.s3.amazonaws.com")+"/"+key
    return f"/runtime/jobs/{job_id}/{path.name}"

@celery_app.task(bind=True)
def process_job(self,job_id:str,inputs:list[str],is_video:bool):
    runtime=Path(os.getenv("RUNTIME_DIR","/app/runtime")); work=runtime/"jobs"/job_id; work.mkdir(parents=True,exist_ok=True)
    try:
        self.update_state(state="PROGRESS",meta={"progress":0.05,"stage":"ingestion"})
        paths=extract_keyframes(inputs[0],work/"frames",target=5) if is_video else inputs
        if not paths: raise ValueError("No usable input images")
        self.update_state(state="PROGRESS",meta={"progress":0.20,"stage":"segmentation_and_measurement"})
        pipeline=DeformationPipeline()
        out=work/"glasses.glb"
        if len(paths)==1:
            result=pipeline.run_from_images(paths[0],output_path=out)
        else:
            result=pipeline.run_from_multiple_images(paths,out)
        self.update_state(state="PROGRESS",meta={"progress":0.90,"stage":"export"})
        url=_upload(out,job_id)
        return {"glb_url":url,"output_glb":str(out),"measurements":result.get("measurements"),"template":result.get("template"),"pipeline":result.get("pipeline")}
    finally:
        # Keep generated job output; remove original upload staging only.
        for p in inputs:
            try: Path(p).unlink(missing_ok=True)
            except OSError: pass
