"""Celery task that reuses the real descriptor-based DeformationPipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2

from .celery_app import celery_app
from backend.pipeline import DeformationPipeline
from backend.video.extract_frames import extract_keyframes
from backend.video.quality_gate import frame_score, is_acceptable

logger = logging.getLogger(__name__)
MAX_RETRIES = 3


def _runtime_dir() -> Path:
    configured = os.getenv("RUNTIME_DIR")
    if configured:
        return Path(configured)
    return Path("/app/runtime") if Path("/app/runtime").exists() else Path("runtime")


def _upload(path: Path, job_id: str) -> str:
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        import boto3

        region = os.getenv("AWS_REGION")
        client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        key = f"models/{job_id}/{path.name}"
        client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "model/gltf-binary"})
        public_base = os.getenv("S3_PUBLIC_BASE")
        if public_base:
            return public_base.rstrip("/") + "/" + key
        if region:
            return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"/runtime/jobs/{job_id}/{path.name}"


def _cleanup_inputs(inputs: list[str]) -> None:
    for raw in inputs:
        try:
            Path(raw).unlink(missing_ok=True)
        except OSError:
            logger.warning("input_cleanup_failed", extra={"path": raw})


@celery_app.task(bind=True, max_retries=MAX_RETRIES, acks_late=True)
def process_job(self, job_id: str, inputs: list[str], is_video: bool):
    runtime = _runtime_dir()
    work = runtime / "jobs" / job_id
    work.mkdir(parents=True, exist_ok=True)

    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.05, "stage": "ingestion"})
        if is_video:
            paths = extract_keyframes(inputs[0], work / "frames", target=5)
        else:
            paths = []
            for raw in inputs:
                image = cv2.imread(raw)
                if image is None:
                    raise ValueError(f"Cannot decode image: {raw}")
                if not is_acceptable(image):
                    metrics = frame_score(image)
                    raise ValueError(
                        f"Rejected low-quality input frame {Path(raw).name}: "
                        f"blur={metrics['blur']:.2f}, glare={metrics['glare']:.4f}"
                    )
                paths.append(raw)

        if not paths:
            raise ValueError("No usable input images")

        self.update_state(state="PROGRESS", meta={"progress": 0.20, "stage": "segmentation_and_measurement"})
        pipeline = DeformationPipeline()
        out = work / "glasses.glb"
        if len(paths) == 1:
            result = pipeline.run_from_images(paths[0], output_path=out)
        else:
            result = pipeline.run_from_multiple_images(paths, out)

        measurements = result.get("measurements")
        (work / "measurements.json").write_text(
            json.dumps(measurements, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.update_state(state="PROGRESS", meta={"progress": 0.90, "stage": "export"})
        url = _upload(out, job_id)
        _cleanup_inputs(inputs)
        logger.info(
            "job_completed",
            extra={"job_id": job_id, "views": len(paths), "output": str(out)},
        )
        return {
            "glb_url": url,
            "output_glb": str(out),
            "measurements": measurements,
            "template": result.get("template"),
            "pipeline": result.get("pipeline"),
        }
    except Exception as exc:
        logger.exception("job_failed", extra={"job_id": job_id, "retry": self.request.retries})
        if self.request.retries < MAX_RETRIES:
            raise self.retry(exc=exc, countdown=min(60, 2 ** self.request.retries))
        _cleanup_inputs(inputs)
        raise
