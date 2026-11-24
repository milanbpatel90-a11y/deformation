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


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def is_near_duplicate(hash_a: str, hash_b: str, settings: VideoExtractionSettings | None = None) -> bool:
    settings = settings or get_settings()
    return hamming_distance(hash_a, hash_b) <= settings.duplicate_hash_threshold


def filter_duplicates(
    items: list[tuple[str, float]],
    settings: VideoExtractionSettings | None = None,
) -> list[int]:
    """Given a list of (phash, score) in candidate order, return the indices
    to KEEP: for each cluster of near-duplicate hashes, keep only the
    highest-scoring member.
    """
    settings = settings or get_settings()
    n = len(items)
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
            if is_near_duplicate(items[i][0], items[j][0], settings):
                suppressed[j] = True
    kept.sort()
    return kept
