"""Diversity-aware frame ranking and selection.

Targets a configurable total (default 12, per the task brief's "Expected
Output" table: 2 front, 1 front_left_45, 1 front_right_45, 2 left_side,
2 right_side, 1 top, 1 bottom, 1 rear_inside = 11) by filling each view's
own quota first, highest composite-score frame first, so a video that
spends most of its time on one near-identical front angle doesn't crowd out
the side/top/bottom views the deformation pipeline actually needs. Any
budget left after every view's quota is filled (or exhausted) goes to one
"best additional angle" slot: the single highest-scoring remaining frame
from the view with the least existing coverage, breaking ties by score.

Never fabricates: a view with zero accepted candidates simply contributes
zero frames and is reported missing by the pipeline, and the total
returned can land anywhere from 0 up to the target -- it is never padded
with low-quality or duplicate frames just to hit a number.
"""

from __future__ import annotations

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.schemas import TARGET_VIEW_COUNTS, FrameCandidate, ViewLabel


def composite_score(candidate: FrameCandidate) -> float:
    """0-100-ish composite used to rank frames within a view bucket."""
    return (
        candidate.quality.overall_quality * 0.55
        + candidate.detection.confidence * 100.0 * 0.35
        + candidate.view_confidence * 100.0 * 0.10
    )


def select_best_per_view(
    candidates: list[FrameCandidate],
    settings: VideoExtractionSettings | None = None,
) -> tuple[dict[ViewLabel, list[FrameCandidate]], set[int]]:
    """Group accepted candidates by view and select up to each view's
    target quota (score-descending), then fill one "best additional angle"
    slot from whatever scores highest among the leftovers, up to
    `settings.max_total_selected`.

    Returns (selected-by-view, frame_index-set-of-the-additional-slot) so
    the caller can flag which SelectedFrame(s) are the "additional" one
    rather than a named view's own quota.
    """
    settings = settings or get_settings()
    buckets: dict[ViewLabel, list[FrameCandidate]] = {}
    for candidate in candidates:
        if candidate.rejected:
            continue
        candidate.composite_score = round(composite_score(candidate), 2)
        buckets.setdefault(candidate.view, []).append(candidate)
    for items in buckets.values():
        items.sort(key=lambda c: c.composite_score, reverse=True)

    selected: dict[ViewLabel, list[FrameCandidate]] = {}
    used_indices: set[int] = set()
    total = 0

    for view in ViewLabel:
        quota = TARGET_VIEW_COUNTS.get(view.value)
        if quota is None:
            continue  # UNKNOWN has no quota of its own; only reachable via the "additional" slot.
        items = buckets.get(view, [])
        take = items[: min(quota, max(0, settings.max_total_selected - total))]
        if take:
            selected[view] = take
            used_indices.update(c.frame_index for c in take)
            total += len(take)

    additional_indices: set[int] = set()
    if total < settings.max_total_selected:
        leftovers = [
            c
            for view, items in buckets.items()
            for c in items
            if c.frame_index not in used_indices
        ]
        if leftovers:
            leftovers.sort(key=lambda c: c.composite_score, reverse=True)
            best = leftovers[0]
            selected.setdefault(best.view, []).append(best)
            used_indices.add(best.frame_index)
            additional_indices.add(best.frame_index)

    return selected, additional_indices
