"""Shape and material classification for eyewear templates."""

from __future__ import annotations

import cv2
import numpy as np

from backend.models import FrameMaterial, FrameShape


class ShapeClassifier:
    """Classify glasses shape and material from front image."""

    TEMPLATE_MAP: dict[FrameShape, str] = {
        FrameShape.ROUND: "round_metal",
        FrameShape.SQUARE: "square_metal",
        FrameShape.GEOMETRIC: "geometric_metal",
        FrameShape.AVIATOR: "aviator",
        FrameShape.CAT_EYE: "cat_eye",
        FrameShape.RECTANGLE: "rectangle_acetate",
        FrameShape.BROWLINE: "browline",
        FrameShape.RIMLESS: "rimless",
    }

    def classify(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[FrameShape, FrameMaterial, bool]:
        """
        Returns (shape, material, has_nose_pads).
        Uses contour geometry heuristics; replace with CNN for production.
        """
        contour = self._get_main_contour(image, mask)
        if contour is None:
            return FrameShape.GEOMETRIC, FrameMaterial.METAL, True

        shape = self._classify_shape(contour)
        material = self._classify_material(image, contour)
        nose_pads = material == FrameMaterial.METAL and shape != FrameShape.RIMLESS
        return shape, material, nose_pads

    def template_name_for(self, shape: FrameShape, material: FrameMaterial) -> str:
        if shape == FrameShape.RECTANGLE and material == FrameMaterial.ACETATE:
            return "rectangle_acetate"
        if shape in self.TEMPLATE_MAP:
            return self.TEMPLATE_MAP[shape]
        return "geometric_metal"

    def _get_main_contour(
        self, image: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray | None:
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                return max(contours, key=cv2.contourArea)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _classify_shape(self, contour: np.ndarray) -> FrameShape:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / max(hull_area, 1)

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertex_count = len(approx)

        if aspect > 2.2 and solidity > 0.85:
            return FrameShape.AVIATOR
        if vertex_count >= 6 and aspect > 1.3:
            return FrameShape.GEOMETRIC
        if aspect < 1.1 and solidity > 0.9:
            return FrameShape.ROUND
        if 1.1 <= aspect <= 1.4 and vertex_count <= 5:
            return FrameShape.SQUARE
        if aspect > 1.5:
            return FrameShape.RECTANGLE
        return FrameShape.GEOMETRIC

    def _classify_material(self, image: np.ndarray, contour: np.ndarray) -> FrameMaterial:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        pixels = image[mask > 0]
        if len(pixels) == 0:
            return FrameMaterial.METAL

        std = np.std(pixels.astype(np.float32), axis=0).mean()
        brightness = pixels.mean()
        if std < 25 and brightness > 120:
            return FrameMaterial.METAL
        if std > 40:
            return FrameMaterial.ACETATE
        return FrameMaterial.METAL
