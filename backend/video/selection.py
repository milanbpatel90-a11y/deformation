"""Select 5-10 well-spread views from the frames that survived the gate.

Three different problems are solved here, in order:

* **Redundancy.** Six samples a second means a slow orbit spends many frames on
  almost the same pose. Near-duplicates are clustered and only the *best*
  representative of each cluster is kept, so a run of ten near-identical frames
  contributes one view rather than ten.
* **Coverage.** Measurements from one side of the orbit only describe one side.
  Retained frames are therefore spread around the clip rather than being the
  highest-scoring frames, which would cluster wherever the lighting happened to
  be best. Occupied angular sectors and temporal span are reported and enforced.
* **Count.** The production contract is 5-10 views, preferring 8. Fewer usable
  views than ``MIN_VIEWS`` is a hard error; fewer than the target is not.

Frames are compared through a silhouette descriptor. Segmentation is the
expensive step and this stage runs before it, so the silhouette is approximated
for free from the clip itself: a per-pixel temporal median across the sampled
frames estimates the static backdrop and each frame's difference from it is the
moving subject. If that estimate degenerates (a handheld clip where nothing is
static) the descriptor falls back to a downscaled grey thumbnail.

Everything here is deterministic: candidates are visited in a fixed order,
comparisons are exact, and ties break on frame index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.video.frame_extractor import Frame
from backend.video.quality_gate import FrameQuality

LOGGER = logging.getLogger(__name__)

#: Descriptor working resolution and grid.
DESCRIPTOR_LONG_EDGE = 320
DESCRIPTOR_GRID = 12


@dataclass(slots=True)
class SelectedFrame:
    """A frame kept for segmentation and measurement."""

    frame: Frame
    quality: FrameQuality
    descriptor: np.ndarray
    foreground_coverage: float
    sector: int
    nearest_distance: float = 0.0
    duplicate_of: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.frame.index,
            "source_index": self.frame.source_index,
            "timestamp": round(self.frame.timestamp, 3),
            "sector": self.sector,
            "quality_score": round(self.quality.score, 4),
            "sharpness": round(self.quality.metrics.get("laplacian_variance", 0.0), 2),
            "glare_ratio": round(self.quality.metrics.get("glare_ratio", 0.0), 4),
            "foreground_coverage": round(self.foreground_coverage, 4),
            "nearest_descriptor_distance": round(self.nearest_distance, 4),
        }


@dataclass(slots=True)
class SelectionResult:
    selected: list[SelectedFrame]
    candidates: int
    gated_out: int
    dropped_redundant: int
    dropped_soft: int
    target_views: int
    min_views: int
    max_views: int
    sectors: int
    warnings: list[str] = field(default_factory=list)
    descriptor_grid: int = DESCRIPTOR_GRID

    @property
    def coverage(self) -> float:
        """Mean nearest-neighbour descriptor distance; higher means better spread."""
        distances = [item.nearest_distance for item in self.selected]
        return float(np.mean(distances)) if distances else 0.0

    @property
    def occupied_sectors(self) -> int:
        return len({item.sector for item in self.selected})

    @property
    def span_ratio(self) -> float:
        """Fraction of the sampled clip that the selection spans."""
        if len(self.selected) < 2:
            return 0.0
        indices = [item.frame.index for item in self.selected]
        return (max(indices) - min(indices)) / max(max(indices), 1)

    def to_dict(self) -> dict:
        return {
            "selected_count": len(self.selected),
            "candidates": self.candidates,
            "gated_out": self.gated_out,
            "dropped_redundant": self.dropped_redundant,
            "dropped_soft": self.dropped_soft,
            "target_views": self.target_views,
            "min_selected": self.min_views,
            "max_selected": self.max_views,
            "occupied_sectors": self.occupied_sectors,
            "total_sectors": self.sectors,
            "span_ratio": round(self.span_ratio, 4),
            "mean_coverage": round(self.coverage, 4),
            "descriptor_grid": self.descriptor_grid,
            "warnings": list(self.warnings),
            "frames": [item.to_dict() for item in self.selected],
        }


class FrameSelector:
    """Pick a bounded, well-spread, sharp subset of the gated frames."""

    #: Production count contract: 5 minimum, 8 preferred, 10 maximum.
    MIN_VIEWS = 5
    PREFERRED_VIEWS = 8
    MAX_VIEWS = 10
    #: Frames softer than this fraction of the candidate median are dropped, so
    #: the selector adapts to a clip that is uniformly softer than a studio still.
    SHARPNESS_FLOOR_RATIO = 0.45
    #: Descriptor distance below which two frames are treated as the same pose.
    #: Calibrated on a 10 s synthetic orbit (2211 frame pairs): pairs that are
    #: genuinely a different pose start at 0.095, so 0.05 merges same-pose frames
    #: with no false merges measured. It is deliberately conservative -- the
    #: descriptor is noisy where the subject is small, so it catches only a
    #: minority of near-identical pairs. The temporal gap below is what actually
    #: guarantees that a run of consecutive frames cannot fill the selection.
    DUPLICATE_DISTANCE = 0.05
    #: Never take two views closer together in the clip than this many sampled
    #: frames, when enough candidates exist to still reach the target. On a
    #: turntable clip the orbit is monotonic in time, so temporal proximity is a
    #: far more reliable duplicate signal than a descriptor computed from a small,
    #: noisy silhouette.
    MIN_GAP_DIVISOR = 2
    #: Angular sectors used to describe how the selection is distributed.
    ANGULAR_SECTORS = 8
    #: A selection drawn from fewer sectors than this has not sampled the orbit.
    #: Only enforced when the candidate pool is large enough to do better.
    MIN_OCCUPIED_SECTORS = 3
    #: Likewise, the selection should span at least this much of the sampled clip.
    MIN_SPAN_RATIO = 0.35
    #: Quality is a mild tie-breaker, never the primary objective.
    QUALITY_TIE_BREAK = 0.35

    def preselect(
        self,
        frames: list[Frame],
        qualities: list[FrameQuality],
        pool_size: int,
    ) -> SelectionResult:
        """Build the bounded, diverse pool that will be segmented.

        Segmentation is the expensive stage, so the whole clip is screened here
        first and only ``pool_size`` frames go forward. No spread requirement is
        enforced, because this pass is deliberately permissive -- the strict
        contract belongs to the final selection, after unusable frames have been
        removed by the visibility gate.
        """
        return self.select(
            frames,
            qualities,
            min_views=1,
            max_views=max(1, int(pool_size)),
            target_views=max(1, int(pool_size)),
            enforce_spread=False,
        )

    def select(
        self,
        frames: list[Frame],
        qualities: list[FrameQuality],
        min_views: int = MIN_VIEWS,
        max_views: int = MAX_VIEWS,
        target_views: int = PREFERRED_VIEWS,
        index_space: int | None = None,
        enforce_spread: bool = True,
    ) -> SelectionResult:
        if len(frames) != len(qualities):
            raise ValueError("frames and qualities must be the same length")
        if not frames:
            raise ValueError("No frames to select from")

        min_views = max(1, int(min_views))
        max_views = max(min_views, int(max_views))
        target_views = int(min(max(target_views, min_views), max_views))
        # Frame indices refer to the sampled clip, which may be wider than the
        # subset handed in (the final pass receives only the surviving pool), so
        # the sector mapping needs the original span.
        span_frames = int(index_space or len(frames))
        warnings: list[str] = []

        accepted = [i for i, quality in enumerate(qualities) if quality.accepted]
        gated_out = len(frames) - len(accepted)
        if not accepted:
            raise ValueError(
                "Every sampled frame failed the quality gate (blur, glare or exposure). "
                "Re-record with steadier motion, even lighting and no direct reflections."
            )

        kept, dropped_soft = self._apply_sharpness_floor(accepted, qualities, min_views, warnings)
        descriptors, coverages = self._foreground_descriptors([frames[i] for i in kept])
        sectors = self._sectors(span_frames)

        items = [
            SelectedFrame(
                frame=frames[index],
                quality=qualities[index],
                descriptor=descriptors[position],
                foreground_coverage=coverages[position],
                sector=self._sector_for(frames[index].index, span_frames, sectors),
            )
            for position, index in enumerate(kept)
        ]

        # Duplicate removal keeps the best representative of each pose cluster.
        deduped, dropped_redundant = self._deduplicate(items)
        # A gap of span/(2*target) sampled frames keeps the selection spread over
        # the clip instead of letting a run of consecutive frames fill it.
        min_gap = max(1, span_frames // max(target_views * self.MIN_GAP_DIVISOR, 1))
        chosen = self._select_for_coverage(deduped, target_views, min_gap)

        if len(chosen) < min_views:
            raise ValueError(
                f"Only {len(chosen)} usable view(s) remained after quality filtering and "
                f"duplicate removal; at least {min_views} are required. Record a slower, "
                "steadier 360 degree orbit in even light."
            )
        if len(chosen) < target_views:
            warnings.append(
                f"Selected {len(chosen)} views against a target of {target_views}; "
                "too few distinct frames passed screening."
            )

        chosen.sort(key=lambda item: item.frame.index)
        self._annotate_distances(chosen)
        if enforce_spread:
            self._enforce_spread(chosen, len(accepted), sectors, warnings)

        LOGGER.info(
            "Frame selection: %s chosen from %s accepted (%s gated, %s duplicate, %s soft, "
            "%s/%s sectors)",
            len(chosen), len(accepted), gated_out, dropped_redundant, dropped_soft,
            len({item.sector for item in chosen}), sectors,
        )
        return SelectionResult(
            selected=chosen,
            candidates=len(accepted),
            gated_out=gated_out,
            dropped_redundant=dropped_redundant,
            dropped_soft=dropped_soft,
            target_views=target_views,
            min_views=min_views,
            max_views=max_views,
            sectors=sectors,
            warnings=warnings,
        )

    # ── stages ──────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_sharpness_floor(
        accepted: list[int],
        qualities: list[FrameQuality],
        min_views: int,
        warnings: list[str],
    ) -> tuple[list[int], int]:
        """Drop frames far softer than this clip's own median."""
        sharpness = np.array(
            [qualities[i].metrics["laplacian_variance"] for i in accepted], dtype=np.float64
        )
        median = float(np.median(sharpness))
        floor = median * FrameSelector.SHARPNESS_FLOOR_RATIO
        kept = [i for i in accepted if qualities[i].metrics["laplacian_variance"] >= floor]
        if len(kept) < min_views:
            # The clip is uniformly soft; relative filtering would starve the
            # fusion, and the absolute gate has already rejected true blur.
            warnings.append(
                "Sharpness varied little across the clip, so relative soft-frame "
                "filtering was skipped."
            )
            return accepted, 0
        return kept, len(accepted) - len(kept)

    def _foreground_descriptors(
        self, frames: list[Frame]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Approximate each frame's silhouette without running segmentation."""
        small = [self._small_gray(frame.image) for frame in frames]
        if not small:
            return [], []

        shape = small[0].shape
        usable = [gray for gray in small if gray.shape == shape]
        if len(usable) != len(small):
            # Mixed resolutions cannot be stacked; fall back to per-frame thumbs.
            thumbs = [self._thumbnail_descriptor(gray) for gray in small]
            return thumbs, [float(np.count_nonzero(t)) / t.size for t in thumbs]

        stack = np.stack(usable, axis=0).astype(np.float32)
        background = np.median(stack, axis=0)

        descriptors: list[np.ndarray] = []
        coverages: list[float] = []
        for gray in usable:
            difference = cv2.absdiff(gray, background.astype(np.uint8))
            _, foreground = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coverage = float(np.count_nonzero(foreground)) / float(foreground.size)
            descriptors.append(self._thumbnail_descriptor(foreground))
            coverages.append(coverage)

        # A turntable clip leaves a clear rotating subject. If almost nothing
        # changes, the camera was moving with the subject and the median is not a
        # background at all.
        if float(np.median(coverages)) < 0.005:
            LOGGER.debug("Temporal-median foreground was empty; using grey thumbnails")
            thumbs = [self._thumbnail_descriptor(gray) for gray in small]
            return thumbs, [float(np.count_nonzero(t)) / t.size for t in thumbs]
        return descriptors, coverages

    @staticmethod
    def _small_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest <= DESCRIPTOR_LONG_EDGE:
            return gray
        scale = DESCRIPTOR_LONG_EDGE / float(longest)
        return cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _thumbnail_descriptor(mask: np.ndarray) -> np.ndarray:
        """Downsample to a unit-length occupancy grid used as a pose signature."""
        grid = cv2.resize(
            mask.astype(np.float32),
            (DESCRIPTOR_GRID, DESCRIPTOR_GRID),
            interpolation=cv2.INTER_AREA,
        )
        grid = grid / 255.0 if grid.max() > 1.5 else grid
        vector = grid.reshape(-1).astype(np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _sectors(frame_count: int) -> int:
        return max(1, min(FrameSelector.ANGULAR_SECTORS, frame_count))

    @staticmethod
    def _sector_for(index: int, frame_count: int, sectors: int) -> int:
        """Which slice of the clip a frame came from."""
        if frame_count <= 1:
            return 0
        return int(index / frame_count * sectors) % sectors

    def _deduplicate(self, items: list[SelectedFrame]) -> tuple[list[SelectedFrame], int]:
        """Cluster near-identical poses, keeping the highest-quality frame.

        Visiting in descending quality means each cluster survives as its *best*
        frame rather than whichever happened to come first in time. Ties break on
        frame index so the outcome is reproducible.
        """
        if not items:
            return [], 0
        ordered = sorted(items, key=lambda item: (-item.quality.score, item.frame.index))
        representatives: list[SelectedFrame] = []
        dropped = 0
        for item in ordered:
            match = next(
                (
                    representative
                    for representative in representatives
                    if self._distance(item.descriptor, representative.descriptor)
                    < self.DUPLICATE_DISTANCE
                ),
                None,
            )
            if match is None:
                representatives.append(item)
            else:
                item.duplicate_of = match.frame.index
                dropped += 1
        return representatives, dropped

    def _select_for_coverage(
        self, items: list[SelectedFrame], target_views: int, min_gap: int = 1
    ) -> list[SelectedFrame]:
        """Farthest-point sampling with a minimum temporal gap.

        The gap is what stops a run of consecutive near-identical frames from
        filling the selection. It is a hard constraint while enough candidates
        remain and is relaxed only when honouring it would miss the target, so a
        short clip still yields as many views as it can.
        """
        if len(items) <= target_views:
            return list(items)

        # Seed with the highest-quality frame, then repeatedly add the frame
        # furthest from everything already chosen.
        seed = max(
            range(len(items)), key=lambda i: (items[i].quality.score, -items[i].frame.index)
        )
        chosen = [seed]
        remaining = set(range(len(items))) - {seed}
        nearest = {
            i: self._distance(items[i].descriptor, items[seed].descriptor) for i in remaining
        }

        while remaining and len(chosen) < target_views:
            eligible = [
                i
                for i in remaining
                if all(
                    abs(items[i].frame.index - items[c].frame.index) >= min_gap for c in chosen
                )
            ]
            # Relax the gap rather than miss the target on a short clip.
            candidates = eligible or remaining
            best = max(
                candidates,
                key=lambda i: (
                    nearest[i] * (1.0 + self.QUALITY_TIE_BREAK * items[i].quality.score),
                    -items[i].frame.index,
                ),
            )
            chosen.append(best)
            remaining.discard(best)
            for i in remaining:
                nearest[i] = min(
                    nearest[i], self._distance(items[i].descriptor, items[best].descriptor)
                )

        return [items[i] for i in chosen]

    def _enforce_spread(
        self,
        chosen: list[SelectedFrame],
        accepted_count: int,
        sectors: int,
        warnings: list[str],
    ) -> None:
        """Fail loudly when a selection only describes one part of the orbit."""
        required_sectors = min(self.MIN_OCCUPIED_SECTORS, sectors)
        if accepted_count < self.MIN_VIEWS * 2 or required_sectors <= 1:
            # Too few candidates for spread to be a meaningful requirement.
            return
        occupied = len({item.sector for item in chosen})
        span = 0.0
        if len(chosen) >= 2:
            indices = [item.frame.index for item in chosen]
            span = (max(indices) - min(indices)) / max(max(indices), 1)
        if occupied < required_sectors:
            raise ValueError(
                f"Selected views cover only {occupied} of {sectors} angular sectors. The usable "
                "frames come from one part of the video and cannot describe a full orbit. "
                "Record a complete 360 degree turn."
            )
        if span < self.MIN_SPAN_RATIO:
            warnings.append(
                f"Selected views span only {span:.2f} of the clip; the usable frames are "
                "concentrated in one part of the orbit."
            )

    @staticmethod
    def _annotate_distances(items: list[SelectedFrame]) -> None:
        """Record the distance to each frame's closest neighbour.

        The minimum -- not the maximum -- is the meaningful spread measure: it
        is what farthest-point sampling maximises, and it exposes a cluster of
        near-duplicate views that a maximum would hide.
        """
        for position, item in enumerate(items):
            best: float | None = None
            for other_position, other in enumerate(items):
                if position == other_position:
                    continue
                distance = FrameSelector._distance(item.descriptor, other.descriptor)
                best = distance if best is None else min(best, distance)
            item.nearest_distance = best if best is not None else 0.0
