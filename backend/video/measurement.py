"""Perspective-aware measurement extraction for orbit video.

The still-image extractor treats every frame as if it were square-on: it takes the
mask's bounding box, converts with a per-image pixel scale derived from a fixed
reference width, and hands back numbers. That is fine for one hand-picked
product photo. Across an orbit it is wrong, and the failure is visible in the
output -- the same pair of glasses reports lens heights of 31.2, 43.8, 51.8 and
56.0 mm depending on how far the frame had turned, because each view's bounding
box was measured as though it were a front view and the results were then
averaged together.

This module measures in *pixels* first and converts to millimetres once, and it
records where every number came from:

``direct_frontal``
    Measured on a square-on frame. No correction applied.
``perspective_normalized``
    Measured on a turned frame and divided by the cosine of its measured yaw,
    which is legitimate because a yaw rotation compresses horizontal extent by
    ``cos(yaw)`` and leaves vertical extent alone.
``side``
    Measured on a profile: the temple arm is the feature, so temple length and
    bend come from here and the frame width explicitly does not.
``top``
    Measured on a plan view, where rim thickness is visible edge-on.
``inferred``
    Not measured; derived from another dimension. Always reported as such.
``insufficient_evidence``
    The orbit contained no view that can measure this dimension. Reported
    instead of silently substituting a default.

The reference scale is applied exactly once, from the frontal reference width the
orbit geometry established, so a caller-supplied reference propagates to every
dimension coherently instead of per-frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.video.geometry import MaskGeometry, OrbitGeometry, OrbitViewGeometry

#: Provenance labels, in increasing order of how much they should be trusted.
DIRECT_FRONTAL = "direct_frontal"
PERSPECTIVE_NORMALIZED = "perspective_normalized"
SIDE_EVIDENCE = "side"
TOP_EVIDENCE = "top"
INFERRED = "inferred"
INSUFFICIENT = "insufficient_evidence"

#: Trust multipliers applied to a view's quality when it contributes a dimension.
EVIDENCE_WEIGHT = {
    DIRECT_FRONTAL: 1.0,
    PERSPECTIVE_NORMALIZED: 0.75,
    SIDE_EVIDENCE: 0.9,
    TOP_EVIDENCE: 0.8,
}

#: Yaw bands that decide what a view may contribute.
DIRECT_FRONTAL_MAX_YAW = 15.0
LENS_PERSPECTIVE_MAX_YAW = 40.0
FRAME_PERSPECTIVE_MAX_YAW = 45.0
SIDE_MIN_YAW = 55.0
#: Below this the cosine correction explodes and the measurement is meaningless.
MIN_USABLE_COS_YAW = 0.35

#: Dimensions the deformation engine consumes.
DIMENSIONS = (
    "frame_width",
    "lens_width",
    "lens_height",
    "bridge_width",
    "temple_length",
    "temple_curve_angle",
    "rim_thickness",
)

#: Dimensions that a profile genuinely cannot supply, whatever its quality.
SIDE_EXCLUDED = frozenset({"frame_width", "lens_width", "lens_height"})


@dataclass(slots=True)
class DimensionObservation:
    """One view's contribution to one dimension, with its provenance."""

    name: str
    frame_index: int
    view: str
    source: str
    value_px: float
    value_mm: float
    weight: float
    yaw_deg: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "view": self.view,
            "source": self.source,
            "value_mm": round(self.value_mm, 2),
            "value_px": round(self.value_px, 1),
            "yaw_deg": round(self.yaw_deg, 1),
            "weight": round(self.weight, 4),
            "detail": self.detail,
        }


@dataclass(slots=True)
class ViewMeasurement:
    """A single view's measurements plus the provenance of each."""

    frame_index: int
    view: str
    confidence: float
    yaw_deg: float
    cos_yaw: float
    quality_score: float
    geometry: MaskGeometry
    measurements: Measurements
    evidence: dict[str, str] = field(default_factory=dict)
    observations: list[DimensionObservation] = field(default_factory=list)

    def observation_map(self) -> dict[str, DimensionObservation]:
        return {item.name: item for item in self.observations}


@dataclass(slots=True)
class OrbitMeasurement:
    """Every view's measurements, their provenance, and the scale that was used."""

    views: list[ViewMeasurement]
    mm_per_px: float
    scale: dict
    notes: list[str] = field(default_factory=list)

    def provenance(self) -> dict[str, list[dict]]:
        """Per dimension, which views contributed and how each was derived."""
        out: dict[str, list[dict]] = {name: [] for name in DIMENSIONS}
        for view in self.views:
            for observation in view.observations:
                entry = observation.to_dict()
                entry["confidence"] = round(view.confidence, 4)
                out[observation.name].append(entry)
        return out

    def to_dict(self) -> dict:
        return {
            "mm_per_px": round(self.mm_per_px, 6),
            "scale": self.scale,
            "per_dimension": self.provenance(),
            "notes": list(self.notes),
        }


def measure_orbit(
    orbit: OrbitGeometry,
    labels: list[str],
    confidences: list[float],
    quality_scores: list[float],
    reference_width_mm: float,
    default_width_mm: float = 135.0,
    scale_source: str = "extractor_default",
    calibrated: bool = False,
) -> OrbitMeasurement:
    """Measure every usable view in pixels, then convert once.

    ``reference_width_mm`` is the width the frame head is *assumed or known* to
    have. Dividing it by the orbit's frontal reference width gives a single
    millimetres-per-pixel figure that every dimension shares, so the numbers stay
    mutually consistent instead of each view inventing its own scale.
    """
    notes: list[str] = []
    frontal_px = float(orbit.frontal_reference_width)
    if frontal_px <= 0:
        raise ValueError(
            "The orbit produced no measurable frame width, so nothing can be scaled."
        )
    mm_per_px = float(reference_width_mm) / frontal_px

    scale = {
        "mode": "reference_width",
        "reference_width_mm": float(reference_width_mm),
        "source": scale_source,
        "calibrated": bool(calibrated),
        "frontal_reference_width_px": int(frontal_px),
        "mm_per_px": round(mm_per_px, 6),
    }
    if not calibrated:
        notes.append(
            "absolute_scale: uncalibrated -- dimensions are anchored to a "
            f"{reference_width_mm:g} mm reference width, not measured in millimetres"
        )

    views: list[ViewMeasurement] = []
    for index, view in enumerate(orbit.views):
        geometry = view.mask_geometry
        if geometry is None:
            continue
        label = labels[index] if index < len(labels) else "front"
        confidence = confidences[index] if index < len(confidences) else 0.0
        quality = quality_scores[index] if index < len(quality_scores) else 0.5
        views.append(
            _measure_view(view, geometry, label, confidence, quality, mm_per_px)
        )

    if not views:
        raise ValueError("No view in the orbit produced usable mask geometry.")

    missing = _missing_dimensions(views)
    for name in missing:
        notes.append(f"{name}: insufficient_evidence -- no view in this orbit can measure it")
    return OrbitMeasurement(views=views, mm_per_px=mm_per_px, scale=scale, notes=notes)


def _missing_dimensions(views: list[ViewMeasurement]) -> list[str]:
    measured = {
        name
        for view in views
        for name, source in view.evidence.items()
        if source != INSUFFICIENT
    }
    return [name for name in DIMENSIONS if name not in measured]


def _measure_view(
    view: OrbitViewGeometry,
    geometry: MaskGeometry,
    label: str,
    confidence: float,
    quality: float,
    mm_per_px: float,
) -> ViewMeasurement:
    """Derive every dimension this view is entitled to contribute."""
    yaw = view.yaw_deg
    cos_yaw = max(view.cos_yaw, 0.0)
    is_side = label == "side" or yaw >= SIDE_MIN_YAW
    is_top = label == "top"

    observations: list[DimensionObservation] = []
    evidence: dict[str, str] = {}

    def add(name: str, value_px: float | None, source: str, detail: str = "") -> None:
        if value_px is None or value_px <= 0 or not math.isfinite(value_px):
            evidence[name] = INSUFFICIENT
            return
        value_mm = value_px * mm_per_px
        if not math.isfinite(value_mm) or value_mm <= 0:
            evidence[name] = INSUFFICIENT
            return
        evidence[name] = source
        observations.append(
            DimensionObservation(
                name=name,
                frame_index=view.index,
                view=label,
                source=source,
                value_px=float(value_px),
                value_mm=float(value_mm),
                weight=float(max(0.05, quality) * EVIDENCE_WEIGHT.get(source, 0.5)),
                yaw_deg=yaw,
                detail=detail,
            )
        )

    # ── frame width: the frame head, never the temple span ──────────────────
    if is_side:
        # A profile's width is the temple's length, not the frame's width.
        evidence["frame_width"] = INSUFFICIENT
    elif yaw <= DIRECT_FRONTAL_MAX_YAW:
        add("frame_width", float(geometry.lens_cluster_width), DIRECT_FRONTAL)
    elif yaw <= FRAME_PERSPECTIVE_MAX_YAW and cos_yaw >= MIN_USABLE_COS_YAW:
        add(
            "frame_width",
            float(geometry.lens_cluster_width) / cos_yaw,
            PERSPECTIVE_NORMALIZED,
            f"divided by cos({yaw:.0f} deg)",
        )
    else:
        evidence["frame_width"] = INSUFFICIENT

    # ── lens width: from the head minus the bridge, halved ──────────────────
    lens_width_px = None
    if geometry.lens_cluster_width > geometry.bridge_span > 0:
        lens_width_px = (geometry.lens_cluster_width - geometry.bridge_span) / 2.0
    elif geometry.lens_runs:
        lens_width_px = float(min(run[1] - run[0] + 1 for run in geometry.lens_runs))
    if is_side or is_top:
        evidence["lens_width"] = INSUFFICIENT
    elif yaw <= DIRECT_FRONTAL_MAX_YAW and geometry.lens_count >= 2:
        add("lens_width", lens_width_px, DIRECT_FRONTAL)
    elif yaw <= LENS_PERSPECTIVE_MAX_YAW and cos_yaw >= MIN_USABLE_COS_YAW:
        add(
            "lens_width",
            None if lens_width_px is None else lens_width_px / cos_yaw,
            PERSPECTIVE_NORMALIZED,
            f"divided by cos({yaw:.0f} deg)",
        )
    else:
        evidence["lens_width"] = INSUFFICIENT

    # ── lens height: measured on the lens cluster, which yaw does not compress
    #    (the rotation axis is vertical), so the correction is identity and the
    #    figure is the same geometric quantity as a frontal view's.
    if is_side or is_top:
        evidence["lens_height"] = INSUFFICIENT
    elif yaw <= DIRECT_FRONTAL_MAX_YAW:
        add("lens_height", float(geometry.lens_cluster_height), DIRECT_FRONTAL)
    elif yaw <= FRAME_PERSPECTIVE_MAX_YAW:
        add(
            "lens_height",
            float(geometry.lens_cluster_height),
            PERSPECTIVE_NORMALIZED,
            "vertical extent is not compressed by yaw, so no cosine correction applies",
        )
    else:
        evidence["lens_height"] = INSUFFICIENT

    # ── bridge width ────────────────────────────────────────────────────────
    if is_side or is_top:
        evidence["bridge_width"] = INSUFFICIENT
    elif geometry.bridge_span <= 0:
        evidence["bridge_width"] = INSUFFICIENT
    elif yaw <= DIRECT_FRONTAL_MAX_YAW:
        add("bridge_width", float(geometry.bridge_span), DIRECT_FRONTAL)
    elif yaw <= LENS_PERSPECTIVE_MAX_YAW and cos_yaw >= MIN_USABLE_COS_YAW:
        add(
            "bridge_width",
            float(geometry.bridge_span) / cos_yaw,
            PERSPECTIVE_NORMALIZED,
            f"divided by cos({yaw:.0f} deg)",
        )
    else:
        evidence["bridge_width"] = INSUFFICIENT

    # ── temple length and bend: side evidence only ──────────────────────────
    if is_side and geometry.temple_span > 0:
        sin_yaw = max(math.sin(math.radians(min(yaw, 90.0))), 1e-3)
        add(
            "temple_length",
            float(geometry.temple_span) / sin_yaw,
            SIDE_EVIDENCE,
            f"projected span divided by sin({min(yaw, 90.0):.0f} deg)",
        )
    else:
        evidence["temple_length"] = INSUFFICIENT

    if is_side and geometry.temple_span >= 20 and geometry.temple_slope_deg > 0.5:
        evidence["temple_curve_angle"] = SIDE_EVIDENCE
        observations.append(
            DimensionObservation(
                name="temple_curve_angle",
                frame_index=view.index,
                view=label,
                source=SIDE_EVIDENCE,
                value_px=float(geometry.temple_drop_px),
                value_mm=float(geometry.temple_slope_deg),
                weight=float(max(0.05, quality) * EVIDENCE_WEIGHT[SIDE_EVIDENCE]),
                yaw_deg=yaw,
                detail="arm centreline slope over its visible span",
            )
        )
    else:
        evidence["temple_curve_angle"] = INSUFFICIENT

    # ── rim thickness: a plan view sees the rim edge-on ─────────────────────
    if is_top and geometry.lens_cluster_width > 0:
        band = float(geometry.max_column_thickness)
        # A usable plan view shows the rim as a narrow band, not the whole frame.
        if 0 < band <= 0.25 * geometry.lens_cluster_width:
            add("rim_thickness", band, TOP_EVIDENCE, "rim band seen edge-on from above")
        else:
            evidence["rim_thickness"] = INSUFFICIENT
    else:
        evidence["rim_thickness"] = INSUFFICIENT

    measurements = _assemble(view, geometry, observations, mm_per_px)
    return ViewMeasurement(
        frame_index=view.index,
        view=label,
        confidence=confidence,
        yaw_deg=yaw,
        cos_yaw=cos_yaw,
        quality_score=float(quality),
        geometry=geometry,
        measurements=measurements,
        evidence=evidence,
        observations=observations,
    )


def _assemble(
    view: OrbitViewGeometry,
    geometry: MaskGeometry,
    observations: list[DimensionObservation],
    mm_per_px: float,
) -> Measurements:
    """Build a Measurements object from this view's own observations.

    Dimensions this view cannot supply are carried forward as its best available
    estimate so the object stays valid, but they are marked ``insufficient`` in
    the evidence map and the fuser refuses to fuse them from this view. Nothing
    here is allowed to masquerade as a measurement.
    """
    values: dict[str, float] = {}
    for observation in observations:
        values[observation.name] = observation.value_mm

    # Fallbacks exist only so the object validates; provenance gates them out.
    width = values.get("frame_width") or geometry.lens_cluster_width * mm_per_px
    lens_w = values.get("lens_width") or max(28.0, width * 0.37)
    lens_h = values.get("lens_height") or max(20.0, width * 0.35)
    bridge = values.get("bridge_width") or 18.0
    temple = values.get("temple_length") or width * 0.97
    curve = values.get("temple_curve_angle") or 28.0
    rim = values.get("rim_thickness") or 1.2

    return Measurements(
        frame_width=round(max(60.0, width), 1),
        lens_width=round(max(20.0, lens_w), 1),
        lens_height=round(max(15.0, lens_h), 1),
        bridge_width=round(max(8.0, bridge), 1),
        temple_length=round(max(80.0, temple), 1),
        rim_thickness=round(max(0.5, rim), 2),
        material=FrameMaterial.METAL,
        shape=FrameShape.GEOMETRIC,
        nose_pads=True,
        temple_curve_angle=round(min(60.0, max(5.0, curve)), 1),
    )
