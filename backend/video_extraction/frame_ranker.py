"""Diversity-aware frame ranking and selection.

Picks the best frames per view bucket instead of the globally top-N by
score, so a video that spends most of its time on one near-identical
front angle doesn't crowd out the side/top views the deformation pipeline
actually needs.
"""

from __future__ import annotations

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.schemas import FrameCandidate, ViewLabel


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
) -> dict[ViewLabel, list[FrameCandidate]]:
    """Group accepted candidates by view and keep the top
    `max_selected_per_view` of each, score-descending."""
    settings = settings or get_settings()
    buckets: dict[ViewLabel, list[FrameCandidate]] = {}
    for candidate in candidates:
        if candidate.rejected:
            continue
        candidate.composite_score = round(composite_score(candidate), 2)
        buckets.setdefault(candidate.view, []).append(candidate)

    selected: dict[ViewLabel, list[FrameCandidate]] = {}
    for view, items in buckets.items():
        items.sort(key=lambda c: c.composite_score, reverse=True)
        selected[view] = items[: settings.max_selected_per_view]
    return selected
