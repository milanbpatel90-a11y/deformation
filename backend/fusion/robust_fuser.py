"""Robust multi-view measurement fusion using a weighted median and MAD.

The averaging fuser in :mod:`backend.fusion.measurement_fuser` is fine for hand
picked product photos, where every image is known to be clean. An orbit clip is
not that: 20-odd automatically selected frames include glancing angles, partial
occlusion and the occasional frame that passed the gate but still measured
badly. A mean lets one such frame drag the result, because a single 40mm error
moves an average of twenty by 2mm and nothing marks it as suspect.

This fuser is built on two robust statistics instead:

* the **weighted median**, which is unmoved by any minority of bad frames
  however extreme, and
* the **median absolute deviation (MAD)**, which measures spread without itself
  being inflated by the outliers it is meant to find.

Each dimension is fused only from the views that can actually see it. A front
frame cannot measure temple length and a side frame cannot measure frame width,
so those combinations carry weight 0 rather than contributing a fabricated
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.models import FrameMaterial, FrameShape, Measurements

#: Provenance label meaning "no view in this capture could measure it". Kept in
#: step with backend.video.measurement.INSUFFICIENT.
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

#: Views that can measure each dimension, and how much to trust them.
#: A weight of 0 removes the view from that dimension entirely.
VIEW_DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    # Frontal: both lenses, both rims and the bridge are square to the camera.
    "front": {
        "frame_width": 1.00, "lens_width": 1.00, "lens_height": 1.00,
        "bridge_width": 1.00, "rim_thickness": 0.25,
        "temple_length": 0.00, "temple_curve_angle": 0.00,
    },
    # Perspective: foreshortening shortens every frontal dimension, so these
    # views inform shape and rough scale but must not dominate a square-on one.
    # Both lateral labels carry the same weights; the left/right distinction is
    # about which way the frame is turned, not how much to trust it.
    "left_front_perspective": {
        "frame_width": 0.55, "lens_width": 0.45, "lens_height": 0.45,
        "bridge_width": 0.45, "rim_thickness": 0.20,
        "temple_length": 0.35, "temple_curve_angle": 0.30,
    },
    "right_front_perspective": {
        "frame_width": 0.55, "lens_width": 0.45, "lens_height": 0.45,
        "bridge_width": 0.45, "rim_thickness": 0.20,
        "temple_length": 0.35, "temple_curve_angle": 0.30,
    },
    # Side: the temple arm is the only dimension visible; the bounding box of a
    # profile is the temple length, not the frame width.
    "side": {
        "frame_width": 0.00, "lens_width": 0.00, "lens_height": 0.00,
        "bridge_width": 0.00, "rim_thickness": 0.30,
        "temple_length": 1.00, "temple_curve_angle": 1.00,
    },
    # Top: full frame width and rim thickness are visible; lens height is not.
    "top": {
        "frame_width": 0.50, "lens_width": 0.30, "lens_height": 0.10,
        "bridge_width": 0.40, "rim_thickness": 1.00,
        "temple_length": 0.40, "temple_curve_angle": 0.25,
    },
    # Reserved. The one-class mask carries no front/rear evidence, so the
    # classifier never emits "rear"; the row exists so a future part-level mask
    # has somewhere to land rather than silently scoring zero.
    "rear": {
        "frame_width": 0.85, "lens_width": 0.70, "lens_height": 0.70,
        "bridge_width": 0.60, "rim_thickness": 0.20,
        "temple_length": 0.00, "temple_curve_angle": 0.00,
    },
}

#: Dimensions fused numerically, with their rounding precision.
NUMERIC_DIMENSIONS: dict[str, int] = {
    "frame_width": 1,
    "lens_width": 1,
    "lens_height": 1,
    "bridge_width": 1,
    "temple_length": 1,
    "rim_thickness": 2,
    "temple_curve_angle": 1,
}


@dataclass(slots=True)
class ViewObservation:
    """One view's measurement set and how much it should count."""

    view: str
    measurements: Measurements
    weight: float = 1.0
    label: str = ""
    #: Per-dimension provenance, e.g. {"temple_length": "side"}. A dimension
    #: marked "insufficient_evidence" is refused from this view even when the
    #: view label would otherwise allow it, because provenance is measured per
    #: frame rather than assumed from the view class.
    evidence: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return self.label or self.view

    def allows(self, dimension: str) -> bool:
        return self.evidence.get(dimension) != "insufficient_evidence"


@dataclass(slots=True)
class DimensionReport:
    """Audit trail for one fused dimension."""

    name: str
    value: float
    unit: str
    contributors: int
    kept: int
    source: str = "fused"
    median: float = 0.0
    mad: float = 0.0
    robust_sigma: float = 0.0
    spread: float = 0.0
    agreement: float = 0.0
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Which views contributed, and with what provenance.
    provenance: list[dict] = field(default_factory=list)
    #: How many kept observations came from each provenance label.
    evidence_mix: dict[str, int] = field(default_factory=dict)

    def has(self, *sources: str) -> bool:
        return any(self.evidence_mix.get(source, 0) > 0 for source in sources)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "contributors": self.contributors,
            "kept": self.kept,
            "median": round(self.median, 4),
            "mad": round(self.mad, 4),
            "robust_sigma": round(self.robust_sigma, 4),
            "spread": round(self.spread, 4),
            "agreement": round(self.agreement, 4),
            "rejected": list(self.rejected),
            "warnings": list(self.warnings),
            "evidence_mix": dict(self.evidence_mix),
            "provenance": list(self.provenance),
        }


@dataclass(slots=True)
class FusionResult:
    measurements: Measurements
    dimensions: dict[str, DimensionReport]
    view_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    #: Mean quality-derived weight of the views that were handed to the fuser.
    mean_weight: float = 0.0

    #: Deterministic confidence levels.
    LEVEL_HIGH = 0.80
    LEVEL_MEDIUM = 0.60

    def to_dict(self) -> dict:
        return {
            "method": "weighted_median_mad",
            "view_counts": dict(self.view_counts),
            "warnings": list(self.warnings),
            "dimensions": {name: report.to_dict() for name, report in self.dimensions.items()},
        }

    def confidence(
        self,
        scale_assumption: str = "reference_width",
        preferred_views: int = 8,
        scale_calibrated: bool = False,
    ) -> dict:
        """Summarise how much evidence actually backs the fused measurement.

        The score is built from what was measured, not from the fact that the
        pipeline finished:

        * ``measurement_consistency`` -- mean cross-view agreement over the
          dimensions that were *measured*, scaled by the fraction of dimensions
          that had direct evidence at all. A dimension derived from a fallback
          carries no evidence, so it lowers the score instead of being hidden.
        * ``quality_consistency`` -- mean view *trust*, scaled by how close the
          view count is to the preferred 8. Trust folds the frame's quality score
          together with how much of the frame the eyewear occupies, because a
          subject filling a few percent of the picture carries less measurement
          precision than one filling a fifth of it. Per-view trust is in the
          manifest, so the aggregate stays auditable. Five acceptable views are
          real evidence, but less of it than eight.

        ``high`` additionally requires at least the preferred number of views:
        perfect agreement between five views is not the same claim as perfect
        agreement between eight, and the level should not pretend otherwise.
        """
        measured = [report for report in self.dimensions.values() if report.source == "fused"]
        unevidenced = [
            report for report in self.dimensions.values() if report.source != "fused"
        ]

        agreement = float(np.mean([report.agreement for report in measured])) if measured else 0.0
        evidence_coverage = len(measured) / max(len(self.dimensions), 1)
        measurement_consistency = agreement * evidence_coverage

        views = sum(self.view_counts.values())
        adequacy = min(1.0, views / max(preferred_views, 1))
        quality_consistency = float(self.mean_weight) * adequacy
        overall = 0.5 * measurement_consistency + 0.5 * quality_consistency

        level = (
            "high" if overall >= self.LEVEL_HIGH
            else "medium" if overall >= self.LEVEL_MEDIUM
            else "low"
        )
        notes: list[str] = []
        caps: list[str] = []

        def cap(reason: str) -> None:
            caps.append(reason)

        if unevidenced:
            notes.append(
                "insufficient evidence for: "
                + ", ".join(sorted(report.name for report in unevidenced))
            )

        # Geometric diversity: an orbit that only ever saw the front cannot have
        # measured anything a front view cannot see, however good those frames are.
        frontal_views = {"front"}
        if self.view_counts and all(view in frontal_views for view in self.view_counts):
            cap("all selected views are frontal; no perspective or side evidence")
        elif len([v for v in self.view_counts if v not in frontal_views]) == 0:
            cap("selection lacks geometric diversity")

        # Dimension-specific evidence requirements.
        temple = self.dimensions.get("temple_length")
        if temple is not None and temple.source != "fused":
            notes.append("temple_length: insufficient_side_evidence")
            cap("temple length has no side evidence")
        elif temple is not None and not temple.has("side"):
            notes.append("temple_length: no observation came from a side view")
            cap("temple length was not measured from a side view")

        curve = self.dimensions.get("temple_curve_angle")
        if curve is not None and curve.source != "fused":
            notes.append("temple_curve_angle: no side evidence, a default is in use")
            cap("temple curve is defaulted")
        elif curve is not None and not curve.has("side"):
            cap("temple curve was not measured from a side view")

        lens = self.dimensions.get("lens_width")
        lens_height = self.dimensions.get("lens_height")
        for report in (lens, lens_height):
            if report is not None and report.source == "fused" and report.spread > 0.08:
                notes.append(
                    f"{report.name}: views disagree by {report.spread:.0%}; "
                    "lens dimensions are not reliable"
                )
                cap(f"{report.name} disagreement is too large")

        if not scale_calibrated:
            notes.append("absolute_scale: uncalibrated")
            cap("scale is uncalibrated")

        if level == "high" and views < preferred_views:
            cap(f"{views} views is below the preferred {preferred_views}")
        if views < 5:
            notes.append(f"only {views} views contributed")

        rejected = sum(len(report.rejected) for report in self.dimensions.values())
        if rejected:
            notes.append(f"{rejected} outlying view measurement(s) excluded by MAD screening")

        if level == "high" and caps:
            level = "medium"
            notes.append("capped at medium: " + "; ".join(caps))
        elif level == "medium" and len(caps) >= 2:
            # Several independent evidence gaps is a low-confidence result.
            level = "low"
            notes.append("lowered to low: " + "; ".join(caps))

        seen: set[str] = set()
        unique_notes = [n for n in notes if not (n in seen or seen.add(n))]

        return {
            "level": level,
            "score": round(float(overall), 4),
            "selected_views": views,
            "measurement_consistency": round(float(measurement_consistency), 4),
            "quality_consistency": round(float(quality_consistency), 4),
            "scale_assumption": scale_assumption,
            "scale_calibrated": bool(scale_calibrated),
            "measured_dimensions": sorted(report.name for report in measured),
            "insufficient_dimensions": sorted(report.name for report in unevidenced),
            "view_distribution": dict(self.view_counts),
            "caps": caps,
            "notes": unique_notes,
        }


class RobustMeasurementFuser:
    """Fuse per-view measurements with a weighted median and MAD screening."""

    def __init__(self, mad_k: float = 3.0, min_tolerance: float = 0.02) -> None:
        #: Keep values within this many robust sigmas of the weighted median.
        self.mad_k = float(mad_k)
        #: Floor for the rejection band, as a fraction of the median, so that a
        #: zero MAD (a unanimous cluster) does not reject trivial jitter.
        self.min_tolerance = float(min_tolerance)

    def fuse(self, observations: list[ViewObservation]) -> FusionResult:
        if not observations:
            raise ValueError("Cannot fuse an empty list of views")

        warnings: list[str] = []
        view_counts: dict[str, int] = {}
        for observation in observations:
            view_counts[observation.view] = view_counts.get(observation.view, 0) + 1
            if observation.view not in VIEW_DIMENSION_WEIGHTS:
                # A label with no weight row would contribute nothing, silently.
                warnings.append(f"unknown view label '{observation.view}' was ignored")

        dimensions: dict[str, DimensionReport] = {}
        values: dict[str, float] = {}

        for name, precision in NUMERIC_DIMENSIONS.items():
            report = self._fuse_dimension(name, precision, observations)
            dimensions[name] = report
            values[name] = report.value
            warnings.extend(f"{name}: {text}" for text in report.warnings)

        material = self._weighted_vote(observations, "material", FrameMaterial.METAL)
        shape = self._weighted_vote(observations, "shape", FrameShape.GEOMETRIC)
        nose_pads = self._weighted_vote(observations, "nose_pads", True)
        color = self._first_value(observations, "color", "#d9a7a2")
        lens_color = self._first_value(observations, "lens_color", None)
        lens_opacity = self._first_value(observations, "lens_opacity", None)

        measurements = Measurements(
            frame_width=values["frame_width"],
            lens_width=values["lens_width"],
            lens_height=values["lens_height"],
            bridge_width=values["bridge_width"],
            temple_length=values["temple_length"],
            rim_thickness=values["rim_thickness"],
            material=material,
            shape=shape,
            nose_pads=bool(nose_pads),
            temple_curve_angle=values["temple_curve_angle"],
            nose_pad_distance=round(values["bridge_width"] * 0.6, 1) if nose_pads else None,
            nose_pad_angle=15.0 if nose_pads else None,
            nose_pad_height=3.0 if nose_pads else None,
            color=color,
            lens_color=lens_color,
            lens_opacity=lens_opacity,
        )
        return FusionResult(
            measurements=measurements,
            dimensions=dimensions,
            view_counts=view_counts,
            warnings=warnings,
            mean_weight=float(
                np.mean([max(0.0, observation.weight) for observation in observations])
            ),
        )

    # ── numeric fusion ──────────────────────────────────────────────────────
    def _fuse_dimension(
        self,
        name: str,
        precision: int,
        observations: list[ViewObservation],
    ) -> DimensionReport:
        entries: list[tuple[float, float, ViewObservation]] = []
        refused: list[dict] = []
        for observation in observations:
            view_weight = VIEW_DIMENSION_WEIGHTS.get(observation.view, {}).get(name, 0.0)
            if view_weight > 0.0 and not observation.allows(name):
                # The view class permits this dimension but the frame's own
                # measured provenance does not: the orbit could not evidence it.
                refused.append(
                    {
                        "view": observation.view,
                        "label": observation.describe(),
                        "source": observation.evidence.get(name, "insufficient_evidence"),
                    }
                )
                continue
            # Quality scales a view's influence but never resurrects a view that
            # cannot see this dimension.
            weight = view_weight * max(0.0, observation.weight)
            raw = getattr(observation.measurements, name, None)
            if weight <= 0.0 or raw is None:
                continue
            value = float(raw)
            if not np.isfinite(value):
                continue
            entries.append((value, weight, observation))

        report = DimensionReport(name=name, value=0.0, unit="mm", contributors=len(entries),
                                 kept=len(entries))
        if name == "temple_curve_angle":
            report.unit = "deg"

        if not entries:
            # Nothing measured this dimension. A fallback value keeps the
            # engineering model valid, but it is labelled as missing evidence
            # rather than presented as a fusion of observations.
            derived, note = self._derive(name, observations)
            report.value = round(derived, precision)
            report.source = "insufficient_evidence"
            report.median = derived
            report.agreement = 0.0
            report.warnings.append(note)
            if refused:
                report.warnings.append(
                    f"{len(refused)} view(s) were refused for {name} on provenance"
                )
            report.evidence_mix = {INSUFFICIENT_EVIDENCE: len(refused)}
            return report

        report.value, report.median, report.mad, report.robust_sigma, kept, rejected = (
            self._robust_median(entries, precision)
        )
        report.kept = kept
        report.rejected = rejected
        report.spread = report.robust_sigma / max(abs(report.median), 1e-9)
        report.agreement = float(max(0.0, min(1.0, 1.0 - report.spread)))

        rejected_labels = {entry["label"] for entry in rejected}
        report.evidence_mix = {}
        for value, _weight, observation in entries:
            if observation.describe() in rejected_labels:
                continue
            source = observation.evidence.get(name, "unspecified")
            report.evidence_mix[source] = report.evidence_mix.get(source, 0) + 1
            report.provenance.append(
                {
                    "view": observation.view,
                    "label": observation.describe(),
                    "source": source,
                    "value": round(float(value), precision),
                }
            )

        if len(entries) == 1:
            report.warnings.append(
                f"only one view could measure {name}; no cross-check was possible"
            )
        if report.spread > 0.05:
            report.warnings.append(
                f"views disagree on {name} (spread {report.spread:.3f}); treat it as approximate"
            )
        return report

    def _robust_median(
        self,
        entries: list[tuple[float, float, ViewObservation]],
        precision: int,
    ) -> tuple[float, float, float, float, int, list[dict]]:
        """Weighted median, MAD screening, then a weighted median of the survivors."""
        values = np.array([entry[0] for entry in entries], dtype=np.float64)
        weights = np.array([entry[1] for entry in entries], dtype=np.float64)

        median = self._weighted_median(values, weights)
        deviations = np.abs(values - median)
        mad = self._weighted_median(deviations, weights)
        robust_sigma = 1.4826 * mad

        # An unanimous cluster has MAD 0; fall back to a small relative band so
        # that floating-point jitter is not treated as disagreement.
        band = max(self.mad_k * robust_sigma, self.min_tolerance * abs(median), 1e-9)
        keep_mask = deviations <= band

        rejected = [
            {
                "view": entry[2].view,
                "label": entry[2].describe(),
                "value": round(float(entry[0]), precision),
                "deviation": round(float(abs(entry[0] - median)), precision),
                "weight": round(float(entry[1]), 4),
            }
            for entry, keep in zip(entries, keep_mask)
            if not keep
        ]

        if not keep_mask.any():  # defensive: cannot happen with a median in the set
            keep_mask = np.ones_like(keep_mask, dtype=bool)

        final = self._weighted_median(values[keep_mask], weights[keep_mask])
        return (
            round(float(final), precision),
            float(median),
            float(mad),
            float(robust_sigma),
            int(keep_mask.sum()),
            rejected,
        )

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Lower weighted median: the value where cumulative weight reaches half."""
        if values.size == 0:
            return 0.0
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_weights = weights[order]
        total = float(sorted_weights.sum())
        if total <= 0.0:
            return float(np.median(sorted_values))
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, total / 2.0, side="left"))
        return float(sorted_values[min(index, sorted_values.size - 1)])

    def _derive(self, name: str, observations: list[ViewObservation]) -> tuple[float, str]:
        """Fall back to the same rules the averaging fuser used, but say so."""
        frame_width = self._weighted_vote_numeric(observations, "frame_width", 145.0)
        if name == "temple_length":
            return (
                frame_width * 0.97,
                "no side view could measure temple length; estimated from frame width",
            )
        if name == "temple_curve_angle":
            return 28.0, "no side view could measure temple curve; using the 28 degree default"
        if name == "rim_thickness":
            return (
                max(0.8, frame_width * 0.008),
                "no top view could measure rim thickness; estimated from frame width",
            )
        if name == "bridge_width":
            return 18.0, "no frontal view could measure bridge width; using the 18mm default"
        if name == "lens_height":
            return (
                max(20.0, frame_width * 0.35),
                f"no view could measure {name}; estimated from frame width",
            )
        if name == "lens_width":
            return (
                max(28.0, frame_width * 0.37),
                f"no view could measure {name}; estimated from frame width",
            )
        return frame_width, "no view could measure this dimension; estimated from frame width"

    def _weighted_vote_numeric(
        self, observations: list[ViewObservation], name: str, fallback: float
    ) -> float:
        entries = [
            (float(getattr(o.measurements, name)), max(0.0, o.weight) * 0.5)
            for o in observations
            if getattr(o.measurements, name, None) is not None
        ]
        if not entries:
            return fallback
        values = np.array([e[0] for e in entries], dtype=np.float64)
        weights = np.array([e[1] for e in entries], dtype=np.float64)
        if weights.sum() <= 0:
            return float(np.mean(values))
        return self._weighted_median(values, weights)

    # ── categorical fusion ──────────────────────────────────────────────────
    @staticmethod
    def _weighted_vote(observations: list[ViewObservation], name: str, fallback):
        tally: dict[object, float] = {}
        for observation in observations:
            value = getattr(observation.measurements, name, None)
            if value is None:
                continue
            weight = max(0.0, observation.weight)
            if weight <= 0.0:
                continue
            tally[value] = tally.get(value, 0.0) + weight
        if not tally:
            return fallback
        return max(tally.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _first_value(observations: list[ViewObservation], name: str, fallback):
        """Use the highest-weighted view that carries an appearance value."""
        ordered = sorted(observations, key=lambda o: o.weight, reverse=True)
        for observation in ordered:
            value = getattr(observation.measurements, name, None)
            if value is not None:
                return value
        return fallback
