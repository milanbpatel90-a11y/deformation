"""Per-frame quality gate for orbit video: reject blur, glare and bad exposure.

The gate is deliberately cheap and mask-free, because it runs on every sampled
frame *before* the expensive YOLO segmentation pass. That ordering is what makes
an 8-12 s clip affordable: ~48-72 frames are screened with a few OpenCV
reductions, then only the 18-24 survivors are segmented.

Calibration note: texture metrics (Laplacian variance, Tenengrad, edge density)
are scale dependent, so they are always computed on a gray image normalised to
:data:`METRIC_LONG_EDGE`. Intensity metrics (glare, contrast, luminance) are
computed on the full-resolution frame so that a resampling filter cannot average
a blown highlight away before it is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Texture metrics are computed at this long edge so thresholds are stable.
METRIC_LONG_EDGE = 640
#: Specular highlights are measured inside the central crop only, because the
#: product sits centred on a turntable and the background is not the subject.
GLARE_ROI_FRACTION = 0.80


@dataclass(slots=True)
class FrameQuality:
    """Outcome of screening one frame."""

    accepted: bool
    score: float
    metrics: dict[str, float]
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "kept" if self.accepted else "rejected"

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
        }


class FrameQualityGate:
    """Screen individual frames for the failure modes that corrupt measurement.

    Blurred frames inflate silhouette bounding boxes, which biases every
    dimension; glare erases rim edges, which biases rim thickness and lens
    detection. Both must be removed *before* fusion, because a weighted median
    can only reject outliers it can see -- a systematically blurred frame still
    carries a plausible-looking number.
    """

    # ── thresholds ──────────────────────────────────────────────────────────
    #: Calibrated against test_images/*.jpg and synthetic degradations of them.
    #: These absolute gates reject only *distinctly* unusable frames: a mild
    #: 1px blur and a 2% highlight both survive, because orbit video is softer
    #: than a studio still and a slightly soft frame still measures correctly.
    #: Frames that are merely the softest of a good set are dropped later by the
    #: selector's relative sharpness floor, which adapts to the clip at hand.
    MIN_LAPLACIAN_VARIANCE = 60.0
    MIN_TENENGRAD = 300.0
    MIN_CONTRAST = 18.0
    MIN_MEAN_LUMA = 35.0
    MAX_MEAN_LUMA = 232.0
    MAX_GLARE_RATIO = 0.050
    MAX_GLARE_BLOB_RATIO = 0.060
    #: A mask this small is treated as "no eyewear found" rather than a small
    #: subject. It is deliberately near zero: a product filmed from across the
    #: room still measures, and only a genuinely empty mask should be discarded.
    #: Frame trust is scaled by coverage separately, in the pipeline.
    MIN_SUBJECT_COVERAGE = 0.002
    MAX_SUBJECT_COVERAGE = 0.90
    #: A directionally smeared frame keeps a plausible Laplacian variance while
    #: losing all detail along one axis, so it needs its own gate.
    MAX_GRADIENT_ANISOTROPY = 0.90
    GLARE_LEVEL = 250

    # ── eyewear visibility (needs a mask, so it runs after segmentation) ─────
    #: Mask touching the frame border within this margin means the eyewear is
    #: partly outside the picture, so its silhouette is truncated and any width
    #: measured from it is an underestimate.
    BORDER_MARGIN_PX = 3
    #: A frame whose mask breaks into more pieces than this is fragmented, which
    #: is how partial occlusion and failed segmentation show up in a binary mask.
    MAX_MASK_COMPONENTS = 12
    #: The largest connected piece should account for most of the mask.
    MIN_LARGEST_COMPONENT_SHARE = 0.45
    #: Mask area as a fraction of its own bounding box. A near-empty box means the
    #: segmentation produced scattered pixels rather than eyewear.
    MIN_MASK_FILL_RATIO = 0.05
    #: Coverage at which a view is considered to have a full pixel budget on the
    #: rim; below it the visibility score falls away proportionally.
    REFERENCE_COVERAGE = 0.02

    #: Component weights for the 0-1 quality score used to rank and weight views.
    _SCORE_WEIGHTS = {
        "sharpness": 0.35,
        "motion": 0.15,
        "glare": 0.20,
        "exposure": 0.15,
        "contrast": 0.10,
        "coverage": 0.05,
    }

    def evaluate(self, image: np.ndarray, mask: np.ndarray | None = None) -> FrameQuality:
        """Screen one BGR frame. ``mask`` is optional and only adds coverage."""
        metrics = self._measure(image, mask)
        reasons = self._reasons(metrics)
        return FrameQuality(
            accepted=not reasons,
            score=self._score(metrics),
            metrics=metrics,
            reasons=reasons,
        )

    def evaluate_all(
        self, images: list[np.ndarray], masks: list[np.ndarray] | None = None
    ) -> list[FrameQuality]:
        return [
            self.evaluate(image, None if masks is None else masks[index])
            for index, image in enumerate(images)
        ]

    def evaluate_visibility(self, mask: np.ndarray) -> FrameQuality:
        """Screen a *segmented* frame for whether the eyewear is actually usable.

        This is the one gate that cannot run before segmentation, because every
        question it answers -- is the subject too small, is it cut off by the
        frame edge, is it broken into fragments -- is a property of the mask. It
        therefore runs on the bounded candidate pool after segmentation, while
        the mask-free gate above still screens the whole clip cheaply.

        Occlusion is approximated by fragmentation: a binary eyewear mask cannot
        say *what* is in front of the glasses, only that the silhouette has come
        apart. That limitation is deliberate and documented rather than papered
        over with a confident-looking occlusion score.
        """
        metrics = self._visibility_metrics(mask)
        reasons = self._visibility_reasons(metrics)
        coverage = metrics["subject_coverage"]
        # Visibility has no texture to score, so its score is coverage-driven.
        score = float(min(1.0, coverage / max(self.REFERENCE_COVERAGE, 1e-9)))
        if reasons:
            score *= 0.5
        return FrameQuality(accepted=not reasons, score=score, metrics=metrics, reasons=reasons)

    def _visibility_metrics(self, mask: np.ndarray) -> dict[str, float]:
        if mask is None or mask.size == 0:
            return {
                "subject_coverage": 0.0,
                "mask_components": 0.0,
                "largest_component_share": 0.0,
                "mask_fill_ratio": 0.0,
                "border_contact": 1.0,
            }

        binary = (mask > 0).astype(np.uint8)
        height, width = binary.shape[:2]
        coverage = float(np.count_nonzero(binary)) / float(binary.size)

        margin = self.BORDER_MARGIN_PX
        edge = np.concatenate([
            binary[:margin, :].ravel(),
            binary[max(height - margin, 0):, :].ravel(),
            binary[:, :margin].ravel(),
            binary[:, max(width - margin, 0):].ravel(),
        ])
        border = int(np.count_nonzero(edge))

        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int32)
        component_count = float(areas.size)
        largest = float(areas.max()) if areas.size else 0.0
        largest_share = largest / max(float(np.count_nonzero(binary)), 1.0)

        rows = np.flatnonzero(binary.any(axis=1))
        cols = np.flatnonzero(binary.any(axis=0))
        if rows.size and cols.size:
            box_area = int(rows[-1] - rows[0] + 1) * int(cols[-1] - cols[0] + 1)
        else:
            box_area = 0
        fill_ratio = float(np.count_nonzero(binary)) / float(max(box_area, 1))

        return {
            "subject_coverage": coverage,
            "mask_components": component_count,
            "largest_component_share": largest_share,
            "mask_fill_ratio": fill_ratio,
            "border_contact": float(border),
        }

    def _visibility_reasons(self, metrics: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        coverage = metrics["subject_coverage"]
        if coverage < self.MIN_SUBJECT_COVERAGE:
            reasons.append(f"no eyewear detected ({coverage:.4f} of frame)")
        elif coverage > self.MAX_SUBJECT_COVERAGE:
            reasons.append(f"eyewear fills the frame ({coverage:.3f}); part of it is cropped")
        if metrics["border_contact"] > 0:
            reasons.append("eyewear touches the frame edge and is partly outside the picture")
        if metrics["mask_components"] > self.MAX_MASK_COMPONENTS:
            reasons.append(f"segmentation fragmented into {metrics['mask_components']:.0f} pieces")
        if 0.0 < metrics["largest_component_share"] < self.MIN_LARGEST_COMPONENT_SHARE:
            reasons.append(
                f"largest eyewear region is only {metrics['largest_component_share']:.2f} of the mask"
            )
        if 0.0 < metrics["mask_fill_ratio"] < self.MIN_MASK_FILL_RATIO:
            reasons.append(
                f"mask is scattered ({metrics['mask_fill_ratio']:.3f} of its bounding box)"
            )
        return reasons

    # ── metrics ─────────────────────────────────────────────────────────────
    def _measure(self, image: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
        if image is None or image.size == 0:
            raise ValueError("Cannot evaluate an empty frame")

        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self._normalise(gray_full)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.Canny(gray, 60, 160)

        energy_x = float(np.mean(grad_x**2))
        energy_y = float(np.mean(grad_y**2))
        total_energy = energy_x + energy_y

        metrics = {
            "laplacian_variance": float(laplacian.var()),
            "tenengrad": float(np.mean(grad_x**2 + grad_y**2)),
            "contrast": float(gray_full.std()),
            "mean_luma": float(gray_full.mean()),
            "edge_density": float(np.count_nonzero(edges)) / float(edges.size),
            "gradient_anisotropy": abs(energy_x - energy_y) / total_energy if total_energy > 0 else 0.0,
        }
        metrics.update(self._glare_metrics(gray_full))

        if mask is not None and mask.size:
            coverage = float(np.count_nonzero(mask)) / float(mask.size)
            metrics["subject_coverage"] = coverage
        else:
            metrics["subject_coverage"] = float("nan")
        return metrics

    @staticmethod
    def _normalise(gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest <= METRIC_LONG_EDGE:
            return gray
        scale = METRIC_LONG_EDGE / float(longest)
        return cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _glare_metrics(self, gray_full: np.ndarray) -> dict[str, float]:
        """Specular highlight load inside the central region of interest."""
        height, width = gray_full.shape[:2]
        if GLARE_ROI_FRACTION < 1.0:
            margin_y = int(height * (1.0 - GLARE_ROI_FRACTION) / 2.0)
            margin_x = int(width * (1.0 - GLARE_ROI_FRACTION) / 2.0)
            roi = gray_full[margin_y : height - margin_y, margin_x : width - margin_x]
        else:
            roi = gray_full
        if roi.size == 0:
            roi = gray_full

        clipped = (roi >= self.GLARE_LEVEL).astype(np.uint8)
        total = float(roi.size)
        glare_ratio = float(np.count_nonzero(clipped)) / total

        blob_ratio = 0.0
        if glare_ratio > 0.0:
            count, _, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
            if count > 1:
                # stats[:, cv2.CC_STAT_AREA]; component 0 is the background.
                blob_ratio = float(stats[1:, cv2.CC_STAT_AREA].max()) / total
        return {"glare_ratio": glare_ratio, "glare_blob_ratio": blob_ratio}

    # ── decisions ───────────────────────────────────────────────────────────
    def _reasons(self, metrics: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        laplacian = metrics["laplacian_variance"]
        if laplacian < self.MIN_LAPLACIAN_VARIANCE:
            reasons.append(f"blur (laplacian {laplacian:.0f} < {self.MIN_LAPLACIAN_VARIANCE:.0f})")
        if metrics["tenengrad"] < self.MIN_TENENGRAD:
            reasons.append(f"soft focus (tenengrad {metrics['tenengrad']:.0f} < {self.MIN_TENENGRAD:.0f})")
        if metrics["glare_blob_ratio"] > self.MAX_GLARE_BLOB_RATIO:
            reasons.append(
                f"glare (blob {metrics['glare_blob_ratio']:.3f} > {self.MAX_GLARE_BLOB_RATIO:.3f})"
            )
        elif metrics["glare_ratio"] > self.MAX_GLARE_RATIO:
            reasons.append(f"glare ({metrics['glare_ratio']:.3f} > {self.MAX_GLARE_RATIO:.3f})")
        if metrics["mean_luma"] < self.MIN_MEAN_LUMA:
            reasons.append(f"underexposed (luma {metrics['mean_luma']:.0f} < {self.MIN_MEAN_LUMA:.0f})")
        elif metrics["mean_luma"] > self.MAX_MEAN_LUMA:
            reasons.append(f"overexposed (luma {metrics['mean_luma']:.0f} > {self.MAX_MEAN_LUMA:.0f})")
        if metrics["contrast"] < self.MIN_CONTRAST:
            reasons.append(f"low contrast ({metrics['contrast']:.0f} < {self.MIN_CONTRAST:.0f})")
        # Directional smear: energy survives on one axis only. A frame that is
        # merely low-detail overall is caught by the Laplacian/Tenengrad gates.
        if (metrics["gradient_anisotropy"] > self.MAX_GRADIENT_ANISOTROPY
                and metrics["tenengrad"] >= self.MIN_TENENGRAD):
            reasons.append(
                f"motion blur (anisotropy {metrics['gradient_anisotropy']:.2f} "
                f"> {self.MAX_GRADIENT_ANISOTROPY:.2f})"
            )
        coverage = metrics["subject_coverage"]
        if not np.isnan(coverage):
            if coverage < self.MIN_SUBJECT_COVERAGE:
                reasons.append(f"no subject ({coverage:.3f} of frame)")
            elif coverage > self.MAX_SUBJECT_COVERAGE:
                reasons.append(f"subject fills frame ({coverage:.3f})")
        return reasons

    def _score(self, metrics: dict[str, float]) -> float:
        """Blend metrics into a 0-1 score for ranking and fusion weighting."""
        sharpness = self._ramp(metrics["laplacian_variance"], self.MIN_LAPLACIAN_VARIANCE)
        glare = 1.0 - self._ramp(metrics["glare_blob_ratio"], self.MAX_GLARE_BLOB_RATIO)
        contrast = self._ramp(metrics["contrast"], self.MIN_CONTRAST)
        exposure = self._exposure_score(metrics["mean_luma"])
        coverage = metrics["subject_coverage"]
        if np.isnan(coverage):
            coverage_score = 1.0  # neutral: not measured at this stage
        else:
            coverage_score = self._ramp(coverage, self.MIN_SUBJECT_COVERAGE)

        components = {
            "sharpness": sharpness,
            "motion": self._inverse_ramp(
                metrics["gradient_anisotropy"], self.MAX_GRADIENT_ANISOTROPY
            ),
            "glare": glare,
            "exposure": exposure,
            "contrast": contrast,
            "coverage": coverage_score,
        }
        total_weight = sum(self._SCORE_WEIGHTS.values())
        score = sum(components[key] * weight for key, weight in self._SCORE_WEIGHTS.items())
        return float(max(0.0, min(1.0, score / total_weight)))

    @staticmethod
    def _ramp(value: float, threshold: float) -> float:
        """Return 0 at the threshold, 1 at twice the threshold, clamped."""
        if threshold <= 0:
            return 1.0
        return float(max(0.0, min(1.0, value / (2.0 * threshold))))

    @staticmethod
    def _inverse_ramp(value: float, limit: float) -> float:
        """Return 1 at zero and 0 at ``limit``, clamped (lower is better)."""
        if limit <= 0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - value / limit)))

    def _exposure_score(self, mean_luma: float) -> float:
        """Full marks inside the accepted luminance window, tapering outside."""
        if self.MIN_MEAN_LUMA <= mean_luma <= self.MAX_MEAN_LUMA:
            return 1.0
        if mean_luma < self.MIN_MEAN_LUMA:
            return float(max(0.0, mean_luma / max(self.MIN_MEAN_LUMA, 1e-6)))
        span = max(255.0 - self.MAX_MEAN_LUMA, 1e-6)
        return float(max(0.0, 1.0 - (mean_luma - self.MAX_MEAN_LUMA) / span))
