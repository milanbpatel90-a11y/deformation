"""Shape and material classification for eyewear templates."""

from __future__ import annotations

import cv2
import numpy as np

from backend.models import BridgeType, FrameFamily, FrameMaterial, FrameShape, RimType, StyleClassification


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

    def classify_shape(self, image: np.ndarray, mask: np.ndarray | None = None) -> FrameShape:
        """Public convenience method — returns only the FrameShape."""
        shape, _material, _nose_pads = self.classify(image, mask)
        return shape

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

    def classify_with_confidence(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[FrameShape, FrameMaterial, bool, dict]:
        """
        Same as classify() but also returns a confidence dict with shape/material scores
        and contour geometry metrics used for the decision.
        """
        contour = self._get_main_contour(image, mask)
        if contour is None:
            return FrameShape.GEOMETRIC, FrameMaterial.METAL, True, {"confidence": 0.0, "method": "default"}

        shape = self._classify_shape(contour)
        material = self._classify_material(image, contour)
        nose_pads = material == FrameMaterial.METAL and shape != FrameShape.RIMLESS

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        hull = cv2.convexHull(contour)
        solidity = cv2.contourArea(contour) / max(cv2.contourArea(hull), 1)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        pixels = image[cv2.drawContours(
            np.zeros(image.shape[:2], np.uint8), [contour], -1, 255, -1
        ) > 0]
        px_std = float(np.std(pixels.astype(np.float32), axis=0).mean()) if len(pixels) else 0.0
        px_mean = float(pixels.mean()) if len(pixels) else 0.0

        confidence = {
            "shape": shape.value,
            "material": material.value,
            "confidence": round(min(1.0, solidity * 0.6 + min(aspect / 3.0, 1.0) * 0.4), 3),
            "method": "contour_heuristic",
            "metrics": {
                "aspect_ratio": round(aspect, 3),
                "solidity": round(solidity, 3),
                "vertex_count": len(approx),
                "pixel_std": round(px_std, 1),
                "pixel_mean": round(px_mean, 1),
            },
        }
        return shape, material, nose_pads, confidence

    def classify_style(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> StyleClassification:
        """Return a semantic style profile for template retrieval."""
        contour = self._get_main_contour(image, mask)
        if contour is None:
            return StyleClassification(
                shape=FrameShape.GEOMETRIC,
                material=FrameMaterial.METAL,
                nose_pads=True,
                frame_family=FrameFamily.GEOMETRIC,
                rim_type=RimType.FULL_RIM,
                bridge_type=BridgeType.PAD,
                lens_aspect_ratio=1.2,
                confidence=0.0,
                metrics={"method": "default"},
            )

        shape, material, nose_pads, confidence = self.classify_with_confidence(image, mask)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        family = self._infer_family(shape, aspect)
        rim_type = self._infer_rim_type(shape, material)
        bridge_type = self._infer_bridge_type(shape, material, nose_pads, aspect)
        metrics = dict(confidence.get("metrics", {}))
        metrics["method"] = confidence.get("method", "contour_heuristic")

        return StyleClassification(
            shape=shape,
            material=material,
            nose_pads=nose_pads,
            frame_family=family,
            rim_type=rim_type,
            bridge_type=bridge_type,
            lens_aspect_ratio=round(aspect, 4),
            confidence=float(confidence.get("confidence", 0.0)),
            metrics=metrics,
        )

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
        # High variance + low brightness → patterned plastic/acetate
        if std > 40:
            return FrameMaterial.ACETATE
        # Low variance + high brightness → specular metal
        if std < 25 and brightness > 120:
            return FrameMaterial.METAL
        # Mid variance + lower brightness → matte plastic
        if brightness < 80:
            return FrameMaterial.PLASTIC
        return FrameMaterial.METAL

    @staticmethod
    def _infer_family(shape: FrameShape, aspect_ratio: float) -> FrameFamily:
        if shape == FrameShape.AVIATOR:
            return FrameFamily.AVIATOR
        if shape == FrameShape.ROUND:
            return FrameFamily.ROUND
        if shape == FrameShape.CAT_EYE:
            return FrameFamily.CAT_EYE
        if shape == FrameShape.BROWLINE:
            return FrameFamily.BROWLINE
        if shape == FrameShape.RIMLESS:
            return FrameFamily.RIMLESS
        if shape == FrameShape.RECTANGLE and aspect_ratio > 1.45:
            return FrameFamily.WAYFARER
        if shape == FrameShape.RECTANGLE:
            return FrameFamily.RECTANGLE
        if shape == FrameShape.SQUARE:
            return FrameFamily.SQUARE
        if aspect_ratio > 1.55:
            return FrameFamily.OVERSIZED
        return FrameFamily.GEOMETRIC

    @staticmethod
    def _infer_rim_type(shape: FrameShape, material: FrameMaterial) -> RimType:
        if shape == FrameShape.RIMLESS:
            return RimType.RIMLESS
        if shape == FrameShape.BROWLINE:
            return RimType.SEMI_RIMLESS
        return RimType.FULL_RIM

    @staticmethod
    def _infer_bridge_type(
        shape: FrameShape,
        material: FrameMaterial,
        nose_pads: bool,
        aspect_ratio: float,
    ) -> BridgeType:
        if shape == FrameShape.AVIATOR:
            return BridgeType.DOUBLE
        if material in {FrameMaterial.ACETATE, FrameMaterial.PLASTIC} and aspect_ratio > 1.2:
            return BridgeType.KEYHOLE
        if nose_pads:
            return BridgeType.PAD
        if shape == FrameShape.RIMLESS:
            return BridgeType.STRAIGHT
        return BridgeType.SADDLE
