"""View classification for eyewear images and 360-degree orbit frames.

Two related but separate classifiers live here:

* :meth:`ViewClassifier.classify_view` is the original four-way classifier used
  by the still-image multi-view endpoint. Its decision rules are unchanged.
* :meth:`ViewClassifier.classify_orbit_view` labels orbit frames as ``front``,
  ``left_front_perspective``, ``right_front_perspective``, ``side`` or ``top``,
  and reports a confidence plus the measurements behind the call.

``rear`` is **not** classified, and :data:`ViewClassifier.REAR_AVAILABLE` is
``False`` so the manifest can say so explicitly instead of leaving a caller to
assume the label was merely absent. That is a measured finding, not an omission:
the intended cue was aperture visibility -- on a front view the lens apertures
are open, on a rear view the temple arms project into them -- but the one-class
``eyewear`` model emits a **solid filled** silhouette. ``aperture_openness``
measured 0.000 and ``hole_count`` 0 on all 18 images in ``test_images/``, so the
signal does not exist in this mask representation and no threshold could recover
it. Recovering ``rear`` needs labelled rear-view orbit data, or a part-level mask
that preserves the apertures.

Left versus right perspective *is* reported, but as a stated convention rather
than a pose estimate: the half of the silhouette with more mask area is treated
as the side turned toward the camera, because the nearer lens and rim subtend
more area. ``lateral_skew`` is exposed so the rule can be re-thresholded against
labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from backend.video.geometry import (
    MaskGeometry,
    OrbitGeometry,
    OrbitViewGeometry,
    analyse_mask,
    analyse_orbit,
)

#: Orbit labels, in the order they are reported.
ORBIT_LABELS = (
    "front",
    "left_front_perspective",
    "right_front_perspective",
    "side",
    "top",
)


@dataclass(slots=True)
class OrbitView:
    """Result of classifying one orbit frame."""

    label: str
    confidence: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "evidence": {
                key: (round(value, 4) if isinstance(value, (int, float)) else value)
                for key, value in self.metrics.items()
            },
        }


class ViewClassifier:
    """Classify eyewear images/masks into view angles."""

    #: A silhouette this many times wider than tall is a profile.
    SIDE_ASPECT = 4.0
    #: Below this the silhouette is taller than wide: looking along the temples.
    TOP_ASPECT = 1.05
    #: A thin horizontal band is a plan (top) view of the frame.
    TOP_VERTICAL_FILL = 0.18
    TOP_MIN_ASPECT = 1.6
    #: Bilateral agreement that marks a frame as mirror-symmetric. 0.70 is where
    #: the clearly square-on photos in test_images/ cluster (0.73-0.99).
    FRONTAL_SYMMETRY = 0.70
    PERSPECTIVE_SYMMETRY = 0.45
    #: Minimum |skew| before the lateral sign is treated as unambiguous.
    PERSPECTIVE_SKEW = 0.02
    #: Orbit yaw bands, in degrees, measured from frame-width foreshortening.
    #: A rendered orbit with known yaws reproduces them to within a degree, and
    #: its temple_visibility rises monotonically with |yaw| (0.06 square-on,
    #: 0.24 at 40, 0.41 at 60, 0.67 at 80), so these bands are where the two cues
    #: agree rather than where a single frame looks small.
    FRONT_YAW = 15.0
    SIDE_YAW = 55.0
    #: A frame this many times wider than tall is the frame head seen edge-on.
    EDGE_ASPECT = 6.0
    #: Plan-view signatures: a box that is mostly empty, or a silhouette taller
    #: than it is wide, or a lens cluster that has grown vertically while yaw
    #: stays low (an angled-top view).
    TOP_FILL_MAX = 0.16
    TOP_CLUSTER_ASPECT = 0.75
    TOP_MAX_YAW = 40.0
    #: The mask carries no validated front/rear evidence, so ``rear`` is never
    #: emitted. Reported in the manifest rather than guessed at.
    REAR_AVAILABLE = False

    # ── original still-image classifier (unchanged behaviour) ───────────────
    def classify_view(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify the view angle based on image dimensions and mask shape properties.

        Returns: 'front', 'side', 'top', or 'perspective'
        """
        metrics = self._silhouette(mask)
        if metrics is None:
            return "front"

        aspect_ratio = metrics["aspect_ratio"]
        symmetry = metrics["symmetry"]

        # Classify based on aspect ratio, size, and symmetry
        if aspect_ratio >= self.SIDE_ASPECT:
            return "side"
        elif aspect_ratio < 1.0:
            return "top"
        elif symmetry < 0.30 or aspect_ratio < 1.45:
            return "perspective"
        else:
            return "front"

    # ── orbit classifier ────────────────────────────────────────────────────
    def classify_sequence(self, masks: list[np.ndarray | None]) -> list[OrbitView]:
        """Classify a whole orbit, using each frame's yaw within the clip.

        An orbit is classified as a sequence, not frame by frame, because pose is
        only measurable *relative to the orbit*: the frame head is widest
        square-on and compresses by ``cos(yaw)`` as the product turns, so the
        clip itself supplies the frontal reference. Measuring a rendered orbit
        with known yaws recovers them to within a degree, which is why yaw -- not
        silhouette size -- drives the label here. A single frame carries no
        foreshortening information, so it falls back to lateral skew alone.
        """
        orbit = analyse_orbit(masks)
        results: list[OrbitView] = []
        for view in orbit.views:
            results.append(self._classify_one(view, orbit))
        return results

    def classify_orbit_view(self, image: np.ndarray, mask: np.ndarray) -> OrbitView:
        """Label one orbit frame within :data:`ORBIT_LABELS`.

        Convenience for a single frame; callers holding a whole clip should use
        :meth:`classify_sequence` so the yaw estimate is available.
        """
        return self.classify_sequence([mask])[0]

    def _classify_one(self, view: OrbitViewGeometry, orbit: OrbitGeometry) -> OrbitView:
        geometry = view.mask_geometry
        if geometry is None:
            # An empty mask carries no angle information; callers reject these
            # frames on visibility, so the label only has to be safe.
            return OrbitView(label="front", confidence=0.0, metrics={"empty_mask": 1.0})

        metrics = self._evidence(view, orbit, geometry)
        aspect = geometry.aspect
        yaw = view.yaw_deg
        yaw_known = view.yaw_source in {"foreshortening", "edge_on"}

        # 1. Plan views. Looking down on the frame collapses its vertical extent
        #    or leaves a mostly empty box; an angled-top view keeps full width but
        #    a tall lens cluster while yaw stays low.
        if aspect < self.TOP_ASPECT:
            return self._orbit("top", 0.75, metrics)
        if geometry.bbox_fill <= self.TOP_FILL_MAX:
            return self._orbit("top", 0.8, metrics)
        if geometry.lens_cluster_aspect >= self.TOP_CLUSTER_ASPECT and yaw < self.TOP_MAX_YAW:
            return self._orbit("top", 0.65, metrics)

        # 2. Edge-on profiles: the frame head collapses into a wide thin bar.
        if aspect >= self.EDGE_ASPECT:
            return self._orbit("side", 0.85, metrics)

        # 3. Profiles by measured yaw.
        if yaw_known and yaw >= self.SIDE_YAW:
            margin = min(1.0, (yaw - self.SIDE_YAW) / 25.0)
            return self._orbit("side", 0.75 + 0.2 * margin, metrics)

        # 4. Square-on: needs a resolved pair of rims, not merely a big blob.
        if yaw_known and yaw <= self.FRONT_YAW and geometry.lens_count >= 2:
            margin = 1.0 - (yaw / self.FRONT_YAW)
            symmetry_bonus = 0.15 if geometry.symmetry >= self.FRONTAL_SYMMETRY else 0.0
            return self._orbit("front", 0.7 + 0.15 * margin + symmetry_bonus, metrics)

        # 5. In between: a perspective view. The sign names the side whose
        #    silhouette projects wider, which is the side presented to the camera.
        return self._label_lateral(metrics, geometry, yaw_known)

    def _label_lateral(
        self, metrics: dict[str, float], geometry: MaskGeometry, yaw_known: bool
    ) -> OrbitView:
        """Label a frontal-hemisphere frame as square-on or left/right perspective.

        The convention is stated plainly because it is an assumption: whichever
        half of the silhouette is heavier is treated as the side turned towards
        the camera. ``lateral_skew`` is reported so the decision can be
        re-thresholded once labelled orbit data exists. When yaw is known, a
        frame that is clearly turned is never called ``front`` merely because its
        skew happens to be small -- the sign is still taken, and the ambiguity is
        reported instead.
        """
        metrics = dict(metrics)
        metrics["rear_available"] = 1.0 if self.REAR_AVAILABLE else 0.0
        skew = metrics["lateral_skew"]
        margin = max(0.0, abs(skew) - self.PERSPECTIVE_SKEW)
        confidence = float(min(1.0, 0.45 + margin * 3.0))
        if not yaw_known:
            # No foreshortening evidence: skew is all there is.
            if skew >= self.PERSPECTIVE_SKEW:
                return self._orbit("right_front_perspective", confidence, metrics)
            if skew <= -self.PERSPECTIVE_SKEW:
                return self._orbit("left_front_perspective", confidence, metrics)
            square_on = geometry.symmetry >= self.FRONTAL_SYMMETRY
            return self._orbit("front", 0.75 if square_on else 0.5, metrics)

        metrics["lateral_sign_ambiguous"] = 1.0 if abs(skew) < self.PERSPECTIVE_SKEW else 0.0
        if abs(skew) < self.PERSPECTIVE_SKEW:
            confidence = min(confidence, 0.4)
        label = "right_front_perspective" if skew >= 0.0 else "left_front_perspective"
        return self._orbit(label, confidence, metrics)

    @staticmethod
    def _evidence(
        view: OrbitViewGeometry, orbit: OrbitGeometry, geometry: MaskGeometry
    ) -> dict[str, float]:
        """Everything behind the label, so a wrong call can be argued with."""
        evidence = dict(geometry.to_dict())
        evidence.update(
            {
                "yaw_deg": round(view.yaw_deg, 2),
                "cos_yaw": round(view.cos_yaw, 4),
                "foreshortening": round(view.foreshortening, 4),
                "frontal_reference_width": float(orbit.frontal_reference_width),
                "rear_available": 1.0 if ViewClassifier.REAR_AVAILABLE else 0.0,
            }
        )
        evidence["yaw_source"] = view.yaw_source  # type: ignore[assignment]
        return evidence

    @staticmethod
    def _orbit(label: str, confidence: float, metrics: dict) -> OrbitView:
        return OrbitView(
            label=label,
            confidence=float(max(0.0, min(1.0, confidence))),
            metrics=metrics,  # type: ignore[arg-type]
        )


    # ── shared geometry ─────────────────────────────────────────────────────
    def _silhouette(self, mask: np.ndarray) -> dict[str, float] | None:
        """Measure the frame outline once, for both classifiers."""
        if mask is None or mask.size == 0:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # A segmented front view is commonly several disconnected components
        # (two rims, bridge and arms). Classifying only the largest component
        # makes a full front photo look like a close-up lens or a top view.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[0] + box[2] for box in boxes)
        y1 = max(box[1] + box[3] for box in boxes)
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        if w == 0 or h == 0:
            return None

        aspect_ratio = float(w) / h

        # Calculate horizontal symmetry
        mid_x = x + w // 2
        left_half = mask[y : y + h, x:mid_x]
        right_half = mask[y : y + h, mid_x : x + w]
        left_area = float(np.count_nonzero(left_half))
        right_area = float(np.count_nonzero(right_half))
        max_area = max(left_area, right_area, 1.0)
        area_balance = min(left_area, right_area) / max_area

        # Equal left/right pixel counts also occur in a side profile with
        # horizontal temples. Compare the two halves as mirrored masks so the
        # two-eye arrangement, rather than just its area, identifies a front.
        common_width = min(left_half.shape[1], right_half.shape[1])
        left_compare = left_half[:, -common_width:]
        right_compare = cv2.flip(right_half[:, :common_width], 1)
        overlap = np.count_nonzero((left_compare > 0) & (right_compare > 0))
        bilateral_similarity = 2.0 * overlap / max(
            np.count_nonzero(left_compare) + np.count_nonzero(right_compare), 1.0
        )
        symmetry = min(area_balance, bilateral_similarity)
        # Signed horizontal area skew: positive means the right half is heavier.
        # Used to tell a left perspective from a right one.
        lateral_skew = (right_area - left_area) / max(left_area + right_area, 1.0)

        roi = mask[y : y + h, x : x + w]
        filled = float(np.count_nonzero(roi))
        vertical_fill = filled / float(w * h)
        bbox_area = float(w * h)

        return {
            "x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "aspect_ratio": aspect_ratio,
            "area_balance": float(area_balance),
            "bilateral_similarity": float(bilateral_similarity),
            "symmetry": float(symmetry),
            "lateral_skew": float(lateral_skew),
            "vertical_fill": vertical_fill,
            "edge_thickness": self._edge_thickness(roi),
            "aperture_openness": self._aperture_openness(roi, bbox_area),
            "hole_count": float(self._hole_count(roi)),
        }

    @staticmethod
    def _edge_thickness(roi: np.ndarray) -> float:
        """Mean vertical thickness of the outer columns, normalised by height.

        Temple arms projecting towards the camera thicken the silhouette's outer
        edges; arms folding away leave them thin.
        """
        height, width = roi.shape[:2]
        band = max(1, int(round(width * 0.08)))
        outer = np.concatenate([roi[:, :band], roi[:, width - band :]], axis=1)
        if outer.size == 0 or height == 0:
            return 0.0
        columns = (outer > 0).sum(axis=0).astype(np.float64) / float(height)
        return float(np.mean(columns)) if columns.size else 0.0

    @staticmethod
    def _aperture_openness(roi: np.ndarray, bbox_area: float) -> float:
        """Fraction of the bounding box that is an enclosed background hole.

        Retained as a diagnostic rather than a decision input: the one-class
        ``eyewear`` mask is solid, so this measures 0.000 for every real photo
        tested. It would carry signal for a part-level mask that preserved the
        lens apertures.
        """
        if bbox_area <= 0:
            return 0.0
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0.0
        hierarchy = hierarchy[0]
        hole_area = 0.0
        for index, contour in enumerate(contours):
            # A hole is a contour with a parent (it is enclosed by the frame).
            if hierarchy[index][3] != -1:
                hole_area += float(cv2.contourArea(contour))
        return float(min(1.0, hole_area / bbox_area))

    @staticmethod
    def _hole_count(roi: np.ndarray) -> int:
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0
        return sum(1 for entry in hierarchy[0] if entry[3] != -1)
