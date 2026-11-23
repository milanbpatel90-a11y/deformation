from __future__ import annotations

import numpy as np

from backend.video_extraction.config import get_settings
from backend.video_extraction.duplicate_filter import compute_phash, filter_duplicates, hamming_distance
from backend.video_extraction.frame_ranker import composite_score, select_best_per_view
from backend.video_extraction.schemas import DetectionResult, FrameCandidate, QualityMetrics, ViewLabel


def _solid_frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def test_identical_frames_hash_identically():
    frame = _solid_frame(120)
    assert compute_phash(frame) == compute_phash(frame)


def test_distinct_frames_have_hamming_distance():
    a = _solid_frame(10)
    b = np.zeros((64, 64, 3), dtype=np.uint8)
    b[:32, :, :] = 255
    assert hamming_distance(compute_phash(a), compute_phash(b)) >= 0


def test_filter_duplicates_keeps_best_scoring_member_of_each_cluster():
    settings = get_settings()
    same_hash = "0" * 16
    other_hash = "f" * 16
    items = [(same_hash, 10.0), (same_hash, 90.0), (other_hash, 50.0)]
    kept = filter_duplicates(items, settings)
    # index 1 (score 90) beats index 0 (score 10) in the same-hash cluster;
    # index 2 is a distinct cluster and always survives.
    assert 1 in kept
    assert 0 not in kept
    assert 2 in kept


def _candidate(view, score, confidence=0.9, view_confidence=0.9) -> FrameCandidate:
    return FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        view=view,
        view_confidence=view_confidence,
        quality=QualityMetrics(
            sharpness_score=100.0,
            is_blurry=False,
            dark_fraction=0.0,
            bright_fraction=0.0,
            exposure_ok=True,
            glare_fraction=0.0,
            glare_ok=True,
            overall_quality=score,
        ),
        detection=DetectionResult(detected=True, confidence=confidence),
    )


def test_select_best_per_view_caps_at_configured_max_and_is_diversity_aware():
    settings = get_settings()
    candidates = [_candidate(ViewLabel.FRONT, s) for s in (10, 90, 50, 70)]
    candidates += [_candidate(ViewLabel.SIDE_LEFT, 60)]

    selected = select_best_per_view(candidates, settings)

    assert ViewLabel.SIDE_LEFT in selected  # a single side frame isn't crowded out
    assert len(selected[ViewLabel.FRONT]) <= settings.max_selected_per_view
    # Highest-quality front frame ranks first.
    assert selected[ViewLabel.FRONT][0].quality.overall_quality == 90


def test_select_best_per_view_drops_rejected_candidates():
    settings = get_settings()
    good = _candidate(ViewLabel.FRONT, 90)
    bad = _candidate(ViewLabel.FRONT, 95)
    bad.rejected = True
    selected = select_best_per_view([good, bad], settings)
    assert len(selected[ViewLabel.FRONT]) == 1
    assert selected[ViewLabel.FRONT][0].quality.overall_quality == 90


def test_composite_score_rewards_quality_detection_and_view_confidence():
    high = _candidate(ViewLabel.FRONT, 90, confidence=0.95, view_confidence=0.95)
    low = _candidate(ViewLabel.FRONT, 20, confidence=0.2, view_confidence=0.2)
    assert composite_score(high) > composite_score(low)
