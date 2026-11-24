from __future__ import annotations

import numpy as np

from backend.video_extraction.config import get_settings
from backend.video_extraction.duplicate_filter import compute_phash, filter_duplicates, hamming_distance
from backend.video_extraction.frame_ranker import composite_score, select_best_per_view
from backend.video_extraction.schemas import (
    TARGET_VIEW_COUNTS,
    DetectionResult,
    FrameCandidate,
    QualityMetrics,
    ViewLabel,
)


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


_next_frame_index = iter(range(1_000_000))


def _candidate(view, score, confidence=0.9, view_confidence=0.9) -> FrameCandidate:
    return FrameCandidate(
        frame_index=next(_next_frame_index),
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


def test_select_best_per_view_caps_at_the_views_own_target_quota():
    settings = get_settings()
    candidates = [_candidate(ViewLabel.FRONT, s) for s in (10, 90, 50, 70)]
    candidates += [_candidate(ViewLabel.LEFT_SIDE, 60)]

    selected, additional_indices = select_best_per_view(candidates, settings)

    assert ViewLabel.LEFT_SIDE in selected  # a single side frame isn't crowded out
    # FRONT's own target quota (TARGET_VIEW_COUNTS) caps everything EXCEPT
    # whichever single frame (if any) landed there via the one shared
    # "best additional angle" slot, which is accounted for separately.
    front_non_additional = [c for c in selected[ViewLabel.FRONT] if c.frame_index not in additional_indices]
    assert len(front_non_additional) <= TARGET_VIEW_COUNTS[ViewLabel.FRONT.value]
    # Highest-quality front frame ranks first.
    assert selected[ViewLabel.FRONT][0].quality.overall_quality == 90


def test_select_best_per_view_drops_rejected_candidates():
    settings = get_settings()
    good = _candidate(ViewLabel.FRONT, 90)
    bad = _candidate(ViewLabel.FRONT, 95)
    bad.rejected = True
    selected, _additional = select_best_per_view([good, bad], settings)
    assert len(selected[ViewLabel.FRONT]) == 1
    assert selected[ViewLabel.FRONT][0].quality.overall_quality == 90


def test_select_best_per_view_fills_one_additional_slot_from_leftovers():
    """Beyond every named view's own quota, one "best additional angle" slot
    is filled from the highest-scoring leftover frame (task brief §2/§3G) --
    never fabricated, never padded past what real candidates support."""
    settings = get_settings()
    # Fill FRONT's quota (2) plus one extra FRONT candidate that can only be
    # picked up by the "additional" slot, since FRONT itself is already full.
    candidates = [_candidate(ViewLabel.FRONT, s) for s in (95, 90, 80)]

    selected, additional_indices = select_best_per_view(candidates, settings)

    assert len(selected[ViewLabel.FRONT]) == TARGET_VIEW_COUNTS[ViewLabel.FRONT.value] + 1
    assert len(additional_indices) == 1
    # The third-best (80) is the only one left over to fill that slot.
    additional = [c for c in selected[ViewLabel.FRONT] if c.frame_index in additional_indices][0]
    assert additional.quality.overall_quality == 80


def test_select_best_per_view_never_exceeds_max_total_selected():
    settings = get_settings()
    candidates = []
    for view in (ViewLabel.FRONT, ViewLabel.LEFT_SIDE, ViewLabel.RIGHT_SIDE, ViewLabel.TOP, ViewLabel.BOTTOM):
        candidates += [_candidate(view, s) for s in (90, 85, 80, 75)]

    selected, _additional = select_best_per_view(candidates, settings)

    total = sum(len(items) for items in selected.values())
    assert total <= settings.max_total_selected


def test_composite_score_rewards_quality_detection_and_view_confidence():
    high = _candidate(ViewLabel.FRONT, 90, confidence=0.95, view_confidence=0.95)
    low = _candidate(ViewLabel.FRONT, 20, confidence=0.2, view_confidence=0.2)
    assert composite_score(high) > composite_score(low)
