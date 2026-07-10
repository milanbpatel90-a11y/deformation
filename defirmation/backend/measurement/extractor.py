"""Extract physical measurements from glasses product images."""

from __future__ import annotations

import cv2
import numpy as np

from backend.models import FrameMaterial, FrameShape, LensContour, Measurements


class MeasurementExtractor:
    """
    Extract mm measurements from front + side images.

  Assumes a reference scale: front image spans known frame width region.
  For MVP, uses pixel ratios calibrated against typical product photo framing.
    """

    # Typical product photo: frame occupies ~70% of image width
    DEFAULT_FRAME_WIDTH_MM = 135.0
    PIXEL_TO_MM_RATIO = None  # computed per image

    def extract_from_images(
        self,
        front: np.ndarray,
        side: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        shape: FrameShape = FrameShape.GEOMETRIC,
        material: FrameMaterial = FrameMaterial.METAL,
        nose_pads: bool = True,
        color: str = "#d9a7a2",
    ) -> tuple[Measurements, LensContour]:
        front_mask = mask if mask is not None else self._auto_mask(front)
        contour = self._largest_contour(front_mask)
        if contour is None:
            return self._default_measurements(shape, material, nose_pads, color), LensContour()

        x, y, w, h = cv2.boundingRect(contour)
        mm_per_px = self.DEFAULT_FRAME_WIDTH_MM / max(w, 1)

        # Detect lens regions via horizontal valley in upper half
        lens_w_px, lens_h_px, bridge_w_px = self._detect_lens_regions(front_mask, x, y, w, h)

        frame_width = w * mm_per_px
        lens_width = lens_w_px * mm_per_px
        lens_height = lens_h_px * mm_per_px
        bridge_width = bridge_w_px * mm_per_px

        temple_length = frame_width * 0.97
        temple_curve_angle = 28.0
        if side is not None:
            temple_length, temple_curve_angle = self._measure_temple(side, mm_per_px)

        rim_thickness = max(0.8, frame_width * 0.008)

        measurements = Measurements(
            frame_width=round(frame_width, 1),
            lens_width=round(lens_width, 1),
            lens_height=round(lens_height, 1),
            bridge_width=round(bridge_width, 1),
            temple_length=round(temple_length, 1),
            rim_thickness=round(rim_thickness, 2),
            material=material,
            shape=shape,
            nose_pads=nose_pads,
            temple_curve_angle=round(temple_curve_angle, 1),
            nose_pad_distance=round(bridge_width * 0.6, 1) if nose_pads else None,
            nose_pad_angle=15.0 if nose_pads else None,
            nose_pad_height=3.0 if nose_pads else None,
            color=color,
        )

        lens_contour = self._extract_lens_contour(front, front_mask, x, y, w, h)
        return measurements, lens_contour

    def extract_lens_contour(self, front: np.ndarray, mask: np.ndarray | None = None) -> LensContour:
        front_mask = mask if mask is not None else self._auto_mask(front)
        contour = self._largest_contour(front_mask)
        if contour is None:
            return LensContour()
        x, y, w, h = cv2.boundingRect(contour)
        return self._extract_lens_contour(front, front_mask, x, y, w, h)

    def _default_measurements(
        self,
        shape: FrameShape,
        material: FrameMaterial,
        nose_pads: bool,
        color: str,
    ) -> Measurements:
        return Measurements(
            frame_width=145.0,
            lens_width=54.0,
            lens_height=50.0,
            bridge_width=18.0,
            temple_length=140.0,
            rim_thickness=1.2,
            material=material,
            shape=shape,
            nose_pads=nose_pads,
            color=color,
        )

    def _auto_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _detect_lens_regions(
        self, mask: np.ndarray, x: int, y: int, w: int, h: int
    ) -> tuple[float, float, float]:
        roi = mask[y : y + h, x : x + w]
        col_sum = roi.sum(axis=0).astype(np.float32)
        mid = w // 2
        bridge_w_px = max(8.0, w * 0.12)
        lens_w_px = (w - bridge_w_px) / 2.0
        lens_h_px = h * 0.75

        # Refine bridge from column minimum near center
        search_start = int(mid - w * 0.15)
        search_end = int(mid + w * 0.15)
        if search_end > search_start:
            center_slice = col_sum[search_start:search_end]
            if len(center_slice) > 0:
                bridge_center = search_start + int(np.argmin(center_slice))
                bridge_w_px = max(bridge_w_px, w * 0.1)
                lens_w_px = (w - bridge_w_px) / 2.0

        return lens_w_px, lens_h_px, bridge_w_px

    def _measure_temple(self, side: np.ndarray, mm_per_px: float) -> tuple[float, float]:
        gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 140.0, 28.0

        longest = max(contours, key=lambda c: cv2.arcLength(c, False))
        length_px = cv2.arcLength(longest, False)
        temple_length = length_px * mm_per_px * 0.85

        if len(longest) >= 3:
            pts = longest.reshape(-1, 2).astype(np.float32)
            vec = pts[-1] - pts[0]
            angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
            curve_angle = min(45.0, max(10.0, angle))
        else:
            curve_angle = 28.0

        return temple_length, curve_angle

    def _extract_lens_contour(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> LensContour:
        """Extract left/right lens contours from the binary mask."""
        if mask is None or mask.sum() == 0:
            return LensContour(left=self._default_lens_polygon("left"), right=self._default_lens_polygon("right"))

        h, w = mask.shape[:2]
        left_mask = mask[:, : w // 2]
        right_mask = mask[:, w // 2 :]

        def contour_from_half(mask_roi: np.ndarray, offset_x: int, half_width: int) -> list[list[float]]:
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = []
            for pt in approx.reshape(-1, 2):
                px = int(pt[0] + offset_x)
                py = int(pt[1])
                if offset_x == 0:
                    x_norm = (px / max(half_width, 1)) * 0.5
                else:
                    x_norm = 0.5 + ((px - offset_x) / max(half_width, 1)) * 0.5
                y_norm = py / max(h, 1)
                points.append([round(x_norm, 4), round(y_norm, 4)])
            return points

        left = contour_from_half(left_mask, 0, w // 2)
        right = contour_from_half(right_mask, w // 2, w - w // 2)

        if not left or not right:
            return LensContour(left=self._default_lens_polygon("left"), right=self._default_lens_polygon("right"))

        return LensContour(left=left, right=right)

    def _contour_to_polygon(
        self,
        edges: np.ndarray,
        offset_x: int,
        offset_y: int,
        frame_w: int,
        frame_h: int,
        side: str,
        n_vertices: int = 8,
    ) -> list[list[float]]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._default_lens_polygon(side)

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        while len(approx) > n_vertices:
            epsilon *= 1.5
            approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 4:
            return self._default_lens_polygon(side)

        points = []
        for pt in approx:
            px = (pt[0][0] + offset_x - offset_x) / max(edges.shape[1], 1)
            py = (pt[0][1] + offset_y - offset_y) / max(frame_h, 1)
            if side == "left":
                px = px * 0.5
            else:
                px = 0.5 + px * 0.5
            points.append([round(px, 4), round(py, 4)])

        return points[:n_vertices]

    def _default_lens_polygon(self, side: str) -> list[list[float]]:
        if side == "left":
            return [
                [0.05, 0.2], [0.2, 0.05], [0.4, 0.05], [0.45, 0.3],
                [0.45, 0.7], [0.4, 0.95], [0.2, 0.95], [0.05, 0.8],
            ]
        return [
            [0.55, 0.05], [0.8, 0.05], [0.95, 0.2], [0.95, 0.8],
            [0.8, 0.95], [0.55, 0.95], [0.5, 0.7], [0.5, 0.3],
        ]
