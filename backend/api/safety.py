"""Shared bounded upload handling and serialized CPU jobs."""
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
_job_lock = Lock()


async def run_job(function, *args, **kwargs):
    # Bound work and protect the shared inference model without blocking the event loop.
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(503, "Another job is running. Please retry shortly.", headers={"Retry-After": "5"})
    try:
        return await run_in_threadpool(function, *args, **kwargs)
    finally:
        _job_lock.release()


def decode_image(contents: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(contents)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds 16 megapixels")
            image.load()
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Invalid or unsupported image") from exc


async def read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds 10 MiB")
    if not contents:
        raise HTTPException(400, "Image upload is empty")
    return contents, await run_in_threadpool(decode_image, contents)
