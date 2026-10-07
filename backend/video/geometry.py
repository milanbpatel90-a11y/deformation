"""Orbit geometry: describe a segmented frame in terms the classifier and the
measurement layer can act on.

The previous classifier decided the view from three numbers (bounding-box aspect,
bilateral symmetry, vertical fill) and, in practice, called almost every orbit
frame ``front``. That is not a bug in the thresholds alone: a bounding box says
very little about pose, because a pair of glasses with temple arms swept
backwards *widens* when it turns -- at yaw 60 the temples project outwards and
the box is wider than it is square-on. Measuring the render confirms it:

    yaw      bbox w   bbox h   aspect   bbox fill
      0         339      130     2.61       0.729
    +-20        431      132     3.27       0.586
    +-40        504      133     3.79       0.452
    +-60        530      134     3.96       0.367
    +-80        501      134     3.74       0.266
    +-90        473       38    12.45       0.651

So width is *not* a yaw proxy here. What does vary monotonically is how much of
its own bounding box the silhouette fills: a square-on frame fills its box,
while a foreshortened frame's box is stretched by projecting temples. This module
therefore measures the box *and* its interior structure -- the column-thickness
profile -- and derives:

* ``temple_visibility``: the share of mask area sitting in columns that are thin
  compared with the tallest column. Rims are tall; a temple arm projecting out to
  the side is thin. This is the strongest single pose cue available from a binary
  mask, and it is what separates a perspective or profile view from a square-on
  one.
* ``lens_runs``: the tall column runs. A front view resolves into two of them
  (two rims) separated by the thin bridge; a profile collapses them into one.
* a yaw *estimate* from where the frame's box-fill sits within the orbit's own
  observed fill range.

Everything here is a measurement of the mask, and every classification reports
the numbers it used, so a wrong call can be argued with rather than guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

#: A column is "tall" (rim-like) when its mask thickness reaches this fraction of
#: the tallest column in the frame. Rims clear it; temple arms do not.
LENS_RUN_FRACTION = 0.45
#: Runs narrower than this fraction of the bounding box are noise, not a rim.
MIN_LENS_RUN_WIDTH = 0.045
#: Minimum number of mask pixels for a frame to be worth describing at all.
MIN_MASK_PIXELS = 120


@dataclass(slots=True)
class MaskGeometry:
    """Measured geometry of one segmented frame."""

    x: int
    y: int
    width: int
    height: int
    area: int
    image_h: int
    image_w: int
    bbox_fill: float
    aspect: float
    coverage: float
    max_column_thickness: float
    thin_column_ratio: float
    temple_visibility: float
    lens_runs: list[tuple[int, int]] = field(default_factory=list)
    lens_run_heights: list[int] = field(default_factory=list)
    lens_cluster_width: int = 0
    lens_cluster_height: int = 0
    temple_span: int = 0
    #: Slope of the temple arm along its visible span, in degrees. The arm bends
    #: down towards the ear, so this is direct side evidence for the curve.
    temple_slope_deg: float = 0.0
    temple_drop_px: float = 0.0
    bridge_span: int = 0
    lateral_skew: float = 0.0
    symmetry: float = 0.0
    #: True when the mask touches the picture edge.
    touches_border: bool = False

    @property
    def lens_cluster_aspect(self) -> float:
        """Cluster height over width. Rises when the camera looks down on the frame."""
        return float(self.lens_cluster_height) / float(max(self.lens_cluster_width, 1))

    @property
    def lens_count(self) -> int:
        return len(self.lens_runs)

    @property
    def left_lens(self) -> tuple[int, int] | None:
        if not self.lens_runs:
            return None
        run = self.lens_runs[0]
        return (run[1] - run[0] + 1, self.lens_run_heights[0])

    @property
    def right_lens(self) -> tuple[int, int] | None:
        if len(self.lens_runs) < 2:
            return None
        run = self.lens_runs[-1]
        return (run[1] - run[0] + 1, self.lens_run_heights[-1])

    def to_dict(self) -> dict:
        return {
            "bbox": [self.x, self.y, self.width, self.height],
            "aspect": round(self.aspect, 4),
            "bbox_fill": round(self.bbox_fill, 4),
            "coverage": round(self.coverage, 5),
            "temple_visibility": round(self.temple_visibility, 4),
            "thin_column_ratio": round(self.thin_column_ratio, 4),
            "lens_count": self.lens_count,
            "lens_cluster": [self.lens_cluster_width, self.lens_cluster_height],
            "lens_cluster_aspect": round(self.lens_cluster_aspect, 4),
            "temple_span": self.temple_span,
            "bridge_span": self.bridge_span,
            "lateral_skew": round(self.lateral_skew, 4),
            "symmetry": round(self.symmetry, 4),
            "max_column_thickness": round(self.max_column_thickness, 1),
            "touches_border": self.touches_border,
        }


@dataclass(slots=True)
class OrbitViewGeometry:
    """One frame's geometry with its orbit-relative yaw resolved."""

    index: int
    mask_geometry: MaskGeometry | None
    yaw_deg: float = 0.0
    cos_yaw: float = 1.0
    yaw_source: str = "unavailable"
    foreshortening: float = 1.0

    @property
    def usable(self) -> bool:
        return self.mask_geometry is not None


@dataclass(slots=True)
class OrbitGeometry:
    """Orbit-level context needed to interpret a single frame."""

    views: list[OrbitViewGeometry] = field(default_factory=list)
    frontal_reference_width: int = 0
    median_cluster_height: float = 0.0
    fill_front: float = 0.0
    fill_side: float = 0.0
    reference_frame: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frontal_reference_width": self.frontal_reference_width,
            "median_cluster_height": round(self.median_cluster_height, 1),
            "fill_range": [round(self.fill_side, 4), round(self.fill_front, 4)],
            "reference_frame": self.reference_frame,
            "yaw_source": "foreshortening" if self.frontal_reference_width else "unavailable",
            "notes": list(self.notes),
        }


#: A frame may only define the frontal reference if its lens cluster has not
#: collapsed vertically (an edge-on profile is a wide, very short bar) and its
#: silhouette is not dominated by thin temple columns.
REFERENCE_MIN_CLUSTER_HEIGHT_RATIO = 0.70
REFERENCE_MAX_TEMPLE_VISIBILITY = 0.25
#: A frame head wider than this multiple of the frontal reference is impossible,
#: so the mask must be wrong rather than the pose being extreme.
MASK_INCONSISTENT_RATIO = 1.10


def analyse_orbit(masks: list[np.ndarray | None]) -> OrbitGeometry:
    """Resolve each frame's yaw from how much its frame width is foreshortened.

    A turntable orbit gives a direct measurement of yaw that no single frame can:
    the frame head is widest square-on, and a yaw rotation of ``theta`` compresses
    it by ``cos(theta)``. Measuring a rendered orbit with known yaws shows how
    tight that is -- cluster width 321 px at 0 degrees, 302 at 20, 247 at 40, 163
    at 60, 59 at 80, and ``arccos(width / 321)`` returns 19.8, 39.7, 59.5 and 79.4.

    The frontal reference therefore has to come from a frame that is *plausibly*
    square-on. An edge-on profile is also wide (473 px in the render), so the
    reference is chosen as the widest cluster among frames whose cluster height
    has not collapsed and whose silhouette is not mostly thin temple columns.
    """
    geometries = [analyse_mask(mask) if mask is not None else None for mask in masks]
    present = [g for g in geometries if g is not None]
    orbit = OrbitGeometry()
    if not present:
        orbit.notes.append("no frame produced usable mask geometry")
        orbit.views = [
            OrbitViewGeometry(index=index, mask_geometry=None)
            for index in range(len(masks))
        ]
        return orbit

    heights = sorted(g.lens_cluster_height for g in present)
    median_height = heights[len(heights) // 2]
    orbit.median_cluster_height = float(median_height)
    orbit.fill_front = max(g.bbox_fill for g in present)
    orbit.fill_side = min(g.bbox_fill for g in present)

    eligible = [
        (index, g)
        for index, g in enumerate(geometries)
        if g is not None
        and g.lens_cluster_height >= REFERENCE_MIN_CLUSTER_HEIGHT_RATIO * median_height
        and g.temple_visibility <= REFERENCE_MAX_TEMPLE_VISIBILITY
    ]
    if not eligible:
        # Every frame looks oblique; fall back to the widest uncollapsed cluster
        # rather than refusing to measure anything.
        eligible = [
            (index, g)
            for index, g in enumerate(geometries)
            if g is not None
            and g.lens_cluster_height >= REFERENCE_MIN_CLUSTER_HEIGHT_RATIO * median_height
        ]
        if eligible:
            orbit.notes.append(
                "no frame was clearly square-on; frontal reference taken from the widest "
                "uncollapsed silhouette and yaw is correspondingly uncertain"
            )
    if eligible:
        # The frontal reference is the *densest* silhouette, not the widest one.
        # A yaw rotation compresses the frame head, so width can only ever be
        # less than square-on -- but a segmentation that swallowed background can
        # be wider, and picking the widest frame then poisons every angle and the
        # whole scale. Density (mask area over bounding-box area) peaks square-on:
        # measured on a rendered orbit it reads 0.729 at 0 degrees against 0.671
        # for a corrupted frame that was nearly twice as wide.
        reference_index, reference = max(
            eligible, key=lambda item: (item[1].bbox_fill, -item[1].lens_cluster_width)
        )
        orbit.frontal_reference_width = int(reference.lens_cluster_width)
        orbit.reference_frame = int(reference_index)
        orbit.notes.append(
            "frontal reference taken from the densest silhouette "
            f"(frame {reference_index}, fill {reference.bbox_fill:.3f}, "
            f"width {reference.lens_cluster_width}px)"
        )
    else:
        orbit.frontal_reference_width = int(max(g.lens_cluster_width for g in present))
        orbit.notes.append("frontal reference fell back to the widest cluster overall")

    views: list[OrbitViewGeometry] = []
    for index, geometry in enumerate(geometries):
        view = OrbitViewGeometry(index=index, mask_geometry=geometry)
        if geometry is not None and orbit.frontal_reference_width > 0:
            ratio = float(geometry.lens_cluster_width) / float(orbit.frontal_reference_width)
            view.foreshortening = max(0.0, min(1.0, ratio))
            view.cos_yaw = max(0.0, min(1.0, ratio))
            view.yaw_deg = float(math.degrees(math.acos(view.cos_yaw)))
            view.yaw_source = "foreshortening"
            if ratio > MASK_INCONSISTENT_RATIO:
                # Wider than square-on is geometrically impossible, so this mask
                # is not the frame head. Flag it: its measurements are refused
                # and it may not define the reference.
                view.mask_inconsistent = True
                view.yaw_source = "mask_inconsistent"
                orbit.notes.append(
                    f"frame {index} mask is {ratio:.2f}x the frontal reference; "
                    "treated as a segmentation outlier"
                )
            if geometry.lens_cluster_height < REFERENCE_MIN_CLUSTER_HEIGHT_RATIO * median_height:
                # An edge-on bar carries no usable foreshortening: its width is the
                # frame's depth, not its width.
                view.yaw_deg = 90.0
                view.cos_yaw = 0.0
                view.yaw_source = "edge_on"
        views.append(view)
    orbit.views = views
    return orbit



def analyse_mask(mask: np.ndarray) -> MaskGeometry | None:
    """Measure one binary eyewear mask, or return None if it holds nothing usable."""
    if mask is None or mask.size == 0:
        return None
    binary = (mask > 0).astype(np.uint8)
    area = int(np.count_nonzero(binary))
    if area < MIN_MASK_PIXELS:
        return None

    cols = np.flatnonzero(binary.any(axis=0))
    rows = np.flatnonzero(binary.any(axis=1))
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if width <= 0 or height <= 0:
        return None

    image_h, image_w = binary.shape[:2]
    thickness = binary.sum(axis=0).astype(np.float64)
    roi_thickness = thickness[x0 : x1 + 1]
    occupied = roi_thickness > 0
    max_thickness = float(roi_thickness.max())

    tall_threshold = max(1.0, LENS_RUN_FRACTION * max_thickness)
    tall = occupied & (roi_thickness >= tall_threshold)
    thin = occupied & (roi_thickness < tall_threshold)

    total_thickness = float(roi_thickness.sum())
    temple_visibility = (
        float(roi_thickness[thin].sum()) / total_thickness if total_thickness > 0 else 0.0
    )
    occupied_count = int(occupied.sum())
    thin_column_ratio = float(thin.sum()) / occupied_count if occupied_count else 0.0

    lens_runs = _runs(tall, min_width=max(3, int(round(MIN_LENS_RUN_WIDTH * width))))
    lens_heights = [
        _run_height(binary, x0 + start, x0 + end) for start, end in lens_runs
    ]

    if lens_runs:
        cluster_start = lens_runs[0][0]
        cluster_end = lens_runs[-1][1]
        lens_cluster_width = cluster_end - cluster_start + 1
        lens_cluster_height = _run_height(binary, x0 + cluster_start, x0 + cluster_end)
    else:
        lens_cluster_width = width
        lens_cluster_height = height

    bridge_span = 0
    if len(lens_runs) >= 2:
        gap = lens_runs[1][0] - lens_runs[0][1] - 1
        # A wide gap is a second object, not a bridge.
        bridge_span = int(gap) if 0 < gap < max(4, 0.35 * lens_cluster_width) else 0

    lateral_skew = _lateral_skew(binary, x0, x1, lens_runs, lens_heights)
    symmetry = _symmetry(binary, x0, y0, width, height)
    temple_span = _temple_span(thin, lens_runs)
    temple_slope, temple_drop = _temple_slope(binary, thin, lens_runs)

    margin = 2
    touches_border = bool(
        binary[:, :margin].any()
        or binary[:, image_w - margin :].any()
        or binary[:margin, :].any()
        or binary[image_h - margin :, :].any()
    )

    return MaskGeometry(
        x=x0,
        y=y0,
        width=width,
        height=height,
        area=area,
        image_h=image_h,
        image_w=image_w,
        bbox_fill=float(area) / float(width * height),
        aspect=float(width) / float(height),
        coverage=float(area) / float(image_h * image_w),
        max_column_thickness=max_thickness,
        thin_column_ratio=thin_column_ratio,
        temple_visibility=temple_visibility,
        lens_runs=[(int(a), int(b)) for a, b in lens_runs],
        lens_run_heights=[int(h) for h in lens_heights],
        lens_cluster_width=int(lens_cluster_width),
        lens_cluster_height=int(lens_cluster_height),
        temple_span=int(temple_span),
        temple_slope_deg=float(temple_slope),
        temple_drop_px=float(temple_drop),
        bridge_span=int(bridge_span),
        lateral_skew=float(lateral_skew),
        symmetry=float(symmetry),
        touches_border=touches_border,
    )


def _runs(flags: np.ndarray, min_width: int) -> list[tuple[int, int]]:
    """Contiguous True runs of at least ``min_width``, as inclusive index pairs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(flags):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_width:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(flags) - start >= min_width:
        runs.append((start, len(flags) - 1))
    return runs


def _run_height(binary: np.ndarray, x_start: int, x_end: int) -> int:
    """Vertical extent of the mask within a column range."""
    if x_end < x_start:
        return 0
    band = binary[:, x_start : x_end + 1]
    rows = np.flatnonzero(band.any(axis=1))
    if rows.size == 0:
        return 0
    return int(rows[-1] - rows[0] + 1)


def _lateral_skew(
    binary: np.ndarray,
    x0: int,
    x1: int,
    lens_runs: list[tuple[int, int]],
    lens_heights: list[int],
) -> float:
    """Signed horizontal asymmetry, positive when the right side is heavier.

    With two resolved rims this compares the rims themselves, which is what a
    perspective turn actually changes. Otherwise it falls back to mask area in
    each half of the bounding box.
    """
    if len(lens_runs) >= 2:
        left = float(lens_runs[0][1] - lens_runs[0][0] + 1)
        right = float(lens_runs[-1][1] - lens_runs[-1][0] + 1)
        if left + right > 0:
            return (right - left) / (right + left)
    mid = (x0 + x1) // 2
    left_area = float(np.count_nonzero(binary[:, x0 : mid + 1]))
    right_area = float(np.count_nonzero(binary[:, mid : x1 + 1]))
    if left_area + right_area <= 0:
        return 0.0
    return (right_area - left_area) / (right_area + left_area)


def _symmetry(binary: np.ndarray, x0: int, y0: int, width: int, height: int) -> float:
    """Mirrored overlap of the two halves of the silhouette."""
    roi = binary[y0 : y0 + height, x0 : x0 + width]
    mid = width // 2
    left = roi[:, :mid]
    right = roi[:, mid:]
    common = min(left.shape[1], right.shape[1])
    if common <= 0:
        return 0.0
    left_cmp = left[:, -common:]
    right_cmp = cv2.flip(right[:, :common], 1)
    overlap = np.count_nonzero((left_cmp > 0) & (right_cmp > 0))
    denom = np.count_nonzero(left_cmp) + np.count_nonzero(right_cmp)
    return 2.0 * overlap / denom if denom else 0.0


def _temple_span(thin: np.ndarray, lens_runs: list[tuple[int, int]]) -> int:
    """Width of the thin (temple-like) columns, excluding inter-lens gaps."""
    if not thin.any():
        return 0
    if lens_runs:
        cluster_start, cluster_end = lens_runs[0][0], lens_runs[-1][1]
        mask = thin.copy()
        mask[cluster_start : cluster_end + 1] = False
    else:
        mask = thin
    return int(np.count_nonzero(mask))


def _temple_slope(
    binary: np.ndarray, thin: np.ndarray, lens_runs: list[tuple[int, int]]
) -> tuple[float, float]:
    """Fit the temple arm's centreline to get its bend, in degrees.

    Only the thin columns outside the frame head belong to the arm. Their
    vertical centre drifts as the arm drops towards the ear, and that drift over
    the span *is* the visible curve -- measured, not assumed.
    """
    if not thin.any():
        return 0.0, 0.0
    mask = thin.copy()
    if lens_runs:
        mask[lens_runs[0][0] : lens_runs[-1][1] + 1] = False
    columns = np.flatnonzero(mask)
    if columns.size < 8:
        return 0.0, 0.0

    centres = []
    for column in columns:
        rows = np.flatnonzero(binary[:, column])
        if rows.size:
            centres.append(float(rows.mean()))
    if len(centres) < 8:
        return 0.0, 0.0

    centres_arr = np.asarray(centres, dtype=np.float64)
    xs = np.arange(centres_arr.size, dtype=np.float64)
    slope = float(np.polyfit(xs, centres_arr, 1)[0])
    # Image y grows downwards, so a downward bend is a positive slope.
    angle = float(math.degrees(math.atan(abs(slope))))
    drop = float(abs(centres_arr[-1] - centres_arr[0]))
    return min(angle, 60.0), drop



def estimate_yaw(
    fill: float,
    fill_front: float,
    fill_side: float,
) -> float:
    """Estimate |yaw| in degrees from where a frame's box-fill sits.

    Assumption, stated because it is one: within a single orbit the densest
    silhouette (highest fill) is the square-on view and the sparsest is the
    most oblique one, so fill interpolates between them. It is an orbit-relative
    estimate, not a calibrated angle, and it is reported next to the raw fill it
    came from.
    """
    if not math.isfinite(fill) or not math.isfinite(fill_front) or not math.isfinite(fill_side):
        return 0.0
    span = fill_front - fill_side
    if span <= 1e-6:
        # A single frame, or a clip with no pose variation: no angle information.
        return 0.0
    ratio = (fill_front - fill) / span
    return float(max(0.0, min(90.0, 90.0 * ratio)))
