"""Near-duplicate frame detection via perceptual hashing (imagehash + Pillow)."""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image

from backend.video_extraction.config import VideoExtractionSettings, get_settings


def compute_phash(frame: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return str(imagehash.phash(pil_image))


def compute_crop_phash(
    frame: np.ndarray, bbox: tuple[int, int, int, int] | None
) -> str | None:
    """Perceptual hash of just the detected eyewear region, not the whole
    frame.

    The task brief asks for duplicate detection to compare both "whole-image
    similarity" and "eyewear-cropped image similarity": a whole-frame phash
    alone is fooled both ways -- a moving/cluttered background can make two
    genuinely-identical views of the glasses look different, while a mostly
    static scene (steady hand, plain background) can make two frames that
    differ only in the glasses' pose look like duplicates. Hashing just the
    bbox crop isolates the object itself from background motion, so the two
    signals catch different failure modes rather than one subsuming the
    other. Returns None (never a fabricated hash) when there is no detection
    bbox to crop, or the bbox is degenerate.
    """
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return compute_phash(crop)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def is_near_duplicate(hash_a: str, hash_b: str, settings: VideoExtractionSettings | None = None) -> bool:
    settings = settings or get_settings()
    return hamming_distance(hash_a, hash_b) <= settings.duplicate_hash_threshold


def _hashes_agree_as_duplicate(
    whole_a: str,
    whole_b: str,
    crop_a: str | None,
    crop_b: str | None,
    settings: VideoExtractionSettings,
) -> bool:
    """Two frames are treated as near-duplicates if EITHER signal says so:
    the whole-frame hash (catches a static scene where the object barely
    moved) OR the cropped-eyewear hash, when both crops are available
    (catches the same object pose surviving a changing background, which the
    whole-frame hash alone would miss). Comparing only whole-frame hashes --
    the previous behavior -- under-detects duplicates whenever the
    background shifts between two otherwise-identical shots of the glasses.
    """
    if is_near_duplicate(whole_a, whole_b, settings):
        return True
    if crop_a is not None and crop_b is not None:
        return is_near_duplicate(crop_a, crop_b, settings)
    return False


def filter_duplicates(
    items: list[tuple[str, float]],
    settings: VideoExtractionSettings | None = None,
    crop_hashes: list[str | None] | None = None,
) -> list[int]:
    """Given a list of (phash, score) in candidate order, return the indices
    to KEEP: for each cluster of near-duplicate hashes, keep only the
    highest-scoring member.

    ``crop_hashes``, when given, is a same-length list of eyewear-cropped
    phashes (or None where no detection bbox was available for that frame)
    compared alongside the whole-frame hash per ``_hashes_agree_as_duplicate``.
    """
    settings = settings or get_settings()
    n = len(items)
    if crop_hashes is not None and len(crop_hashes) != n:
        raise ValueError("crop_hashes must be the same length as items")
    kept: list[int] = []
    suppressed = [False] * n
    # Greedy, score-descending: the best frame in each near-duplicate
    # cluster survives, everything within the hash threshold of it is
    # suppressed.
    order = sorted(range(n), key=lambda i: items[i][1], reverse=True)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            crop_i = crop_hashes[i] if crop_hashes is not None else None
            crop_j = crop_hashes[j] if crop_hashes is not None else None
            if _hashes_agree_as_duplicate(items[i][0], items[j][0], crop_i, crop_j, settings):
                suppressed[j] = True
    kept.sort()
    return kept
