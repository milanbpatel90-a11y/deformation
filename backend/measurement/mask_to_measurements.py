"""
Convert YOLO segmentation masks to parametric measurements for 3D generation.
Extracts precise measurements from multi-class segmentation masks.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from backend.models import Measurements, FrameShape, FrameMaterial, LensContour


class MaskToMeasurements:
    """Extract parametric measurements from segmentation masks."""
    
    # Class IDs from data.yaml
    CLASS_RIM = 0
    CLASS_TEMPLE = 1
    CLASS_BRIDGE = 2
    CLASS_LEFT_LENS = 3
    CLASS_RIGHT_LENS = 4
    CLASS_NOSE_PAD = 5
    
    # Default calibration: assume frame occupies 70% of image width
    DEFAULT_FRAME_WIDTH_MM = 140.0
    
    def __init__(self, reference_width_mm: float = DEFAULT_FRAME_WIDTH_MM):
        """
        Initialize converter.
        
        Args:
            reference_width_mm: Known frame width for calibration
        """
        self.reference_width_mm = reference_width_mm
    
    def extract_from_yolo_results(
        self,
        image: np.ndarray,
        results,
        shape: FrameShape = FrameShape.GEOMETRIC,
        material: FrameMaterial = FrameMaterial.METAL,
        color: str = "#000000"
    ) -> Tuple[Measurements, LensContour]:
        """
        Extract measurements from YOLO segmentation results.
        
        Args:
            image: Original image
            results: YOLO model results
            shape: Frame shape classification
            material: Frame material
            color: Frame color hex
            
        Returns:
            Tuple of (Measurements, LensContour)
        """
        h, w = image.shape[:2]
        masks = self._extract_masks_by_class(results, (h, w))
        return self.extract_from_class_masks(image, masks, shape, material, color)

    def extract_from_class_masks(
        self,
        image: np.ndarray,
        class_masks: Dict[int, np.ndarray],
        shape: FrameShape = FrameShape.GEOMETRIC,
        material: FrameMaterial = FrameMaterial.METAL,
        color: str = "#000000",
    ) -> Tuple[Measurements, LensContour]:
        """Convert full-resolution, class-indexed masks without rerunning YOLO."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a valid BGR image for measurement extraction.")
        image_h, image_w = image.shape[:2]
        masks: Dict[int, np.ndarray] = {}
        for raw_class_id, raw_mask in class_masks.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid eyewear class ID: {raw_class_id!r}") from exc
            if class_id not in {
                self.CLASS_RIM, self.CLASS_TEMPLE, self.CLASS_BRIDGE,
                self.CLASS_LEFT_LENS, self.CLASS_RIGHT_LENS, self.CLASS_NOSE_PAD,
            }:
                raise ValueError(f"Unsupported eyewear part class ID: {class_id}")
            mask = np.asarray(raw_mask)
            if mask.ndim != 2 or mask.shape != (image_h, image_w):
                raise ValueError(
                    f"Class {class_id} mask must have shape {(image_h, image_w)}; "
                    f"received {mask.shape}."
                )
            masks[class_id] = (mask > 0).astype(np.uint8) * 255

        if not any(np.any(mask) for mask in masks.values()):
            raise ValueError("The six-class eyewear model returned no non-empty part masks.")

        # Lens polygons must come from actual left/right lens detections. Do not
        # fabricate default lens shapes when the model failed to segment them.
        missing_lenses = [
            name for class_id, name in (
                (self.CLASS_LEFT_LENS, "left_lens"),
                (self.CLASS_RIGHT_LENS, "right_lens"),
            )
            if class_id not in masks or not np.any(masks[class_id])
        ]
        if missing_lenses:
            raise ValueError(
                "Six-class segmentation is incomplete; missing non-empty "
                + ", ".join(missing_lenses)
                + " mask(s). Review the image/annotations or use another view."
            )

        mm_per_px = self._calculate_scale(masks.get(self.CLASS_RIM), image_w)
        measurements = self._extract_measurements(masks, mm_per_px, shape, material, color)
        lens_contour = self._extract_lens_contours(masks, image_h, image_w)
        return measurements, lens_contour
    
    def _extract_masks_by_class(
        self,
        results,
        image_size: Tuple[int, int]
    ) -> Dict[int, np.ndarray]:
        """
        Extract binary masks for each class from YOLO results.
        
        Returns:
            Dict mapping class_id to binary mask (uint8, 0-255)
        """
        h, w = image_size
        masks = {}
        
        for result in results:
            if result.masks is None:
                continue
            if result.boxes is None:
                raise ValueError("YOLO returned masks without class boxes; cannot assign measurement parts.")

            mask_items = list(result.masks.data)
            box_items = list(result.boxes)
            if len(mask_items) != len(box_items):
                raise ValueError("YOLO returned different numbers of masks and boxes; refusing ambiguous class pairing.")

            for mask_data, box in zip(mask_items, box_items):
                class_id = int(box.cls[0])
                if class_id not in {
                    self.CLASS_RIM, self.CLASS_TEMPLE, self.CLASS_BRIDGE,
                    self.CLASS_LEFT_LENS, self.CLASS_RIGHT_LENS, self.CLASS_NOSE_PAD,
                }:
                    raise ValueError(f"Unexpected eyewear part class ID {class_id}.")

                # Resize mask to image size
                raw_mask = mask_data.cpu().numpy() if hasattr(mask_data, "cpu") else np.asarray(mask_data)
                mask = cv2.resize(
                    raw_mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Convert to binary
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Merge with existing mask for this class
                if class_id in masks:
                    masks[class_id] = cv2.bitwise_or(masks[class_id], binary_mask)
                else:
                    masks[class_id] = binary_mask
        
        return masks
    
    def _calculate_scale(self, rim_mask: Optional[np.ndarray], image_width: int) -> float:
        """
        Calculate mm per pixel ratio from rim mask.
        
        Args:
            rim_mask: Binary mask of frame rim
            image_width: Image width in pixels
            
        Returns:
            Millimeters per pixel
        """
        if rim_mask is None or rim_mask.sum() == 0:
            # This is an assumed scale, not physical calibration. Supply the
            # manufacturer's known frame width for dimensionally reliable output.
            return self.reference_width_mm / (image_width * 0.7)

        points = cv2.findNonZero((rim_mask > 0).astype(np.uint8))
        if points is None:
            return self.reference_width_mm / (image_width * 0.7)
        # Use the bounds of all rim components. Taking only the largest contour
        # would measure one eye/rim, not the total left-to-right frame width.
        x, y, w, h = cv2.boundingRect(points)
        frame_width_px = w
        
        # Calculate scale
        mm_per_px = self.reference_width_mm / max(frame_width_px, 1)
        
        return mm_per_px
    
    def _extract_measurements(
        self,
        masks: Dict[int, np.ndarray],
        mm_per_px: float,
        shape: FrameShape,
        material: FrameMaterial,
        color: str
    ) -> Measurements:
        """Extract all measurements from masks."""
        
        # Frame width (from rim or combined lens masks)
        frame_width = self._measure_frame_width(masks, mm_per_px)
        
        # Lens dimensions
        lens_width, lens_height = self._measure_lens_dimensions(masks, mm_per_px)
        
        # Bridge width
        bridge_width = self._measure_bridge_width(masks, mm_per_px)
        
        # Temple length
        temple_length = self._measure_temple_length(masks, mm_per_px)
        
        # Temple curve angle
        temple_curve_angle = self._measure_temple_curve(masks)
        
        # Nose pads
        has_nose_pads = self.CLASS_NOSE_PAD in masks and masks[self.CLASS_NOSE_PAD].sum() > 0
        nose_pad_distance = bridge_width * 0.6 if has_nose_pads else None
        nose_pad_angle = 15.0 if has_nose_pads else None
        nose_pad_height = 3.0 if has_nose_pads else None
        
        # Rim thickness (estimate from material)
        rim_thickness = 1.2 if material == FrameMaterial.METAL else 2.0
        
        return Measurements(
            frame_width=round(frame_width, 1),
            lens_width=round(lens_width, 1),
            lens_height=round(lens_height, 1),
            bridge_width=round(bridge_width, 1),
            temple_length=round(temple_length, 1),
            rim_thickness=round(rim_thickness, 2),
            material=material,
            shape=shape,
            nose_pads=has_nose_pads,
            temple_curve_angle=round(temple_curve_angle, 1),
            nose_pad_distance=nose_pad_distance,
            nose_pad_angle=nose_pad_angle,
            nose_pad_height=nose_pad_height,
            color=color
        )
    
    def _measure_frame_width(self, masks: Dict[int, np.ndarray], mm_per_px: float) -> float:
        """Measure total frame width."""
        # Prefer a non-empty rim mask. Rimless frames legitimately have no
        # rim pixels, so use the measured lens/bridge envelope in that case.
        rim_mask = masks.get(self.CLASS_RIM)
        if rim_mask is not None and np.any(rim_mask):
            mask = rim_mask
        elif (
            masks.get(self.CLASS_LEFT_LENS) is not None
            and masks.get(self.CLASS_RIGHT_LENS) is not None
            and np.any(masks[self.CLASS_LEFT_LENS])
            and np.any(masks[self.CLASS_RIGHT_LENS])
        ):
            mask = cv2.bitwise_or(masks[self.CLASS_LEFT_LENS], masks[self.CLASS_RIGHT_LENS])
            bridge_mask = masks.get(self.CLASS_BRIDGE)
            if bridge_mask is not None and np.any(bridge_mask):
                mask = cv2.bitwise_or(mask, bridge_mask)
        else:
            return self.reference_width_mm
        
        points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if points is None:
            return self.reference_width_mm

        # The rim is usually two disconnected lens surrounds. Measure the
        # full union bounds rather than only the larger single-eye contour.
        x, y, w, h = cv2.boundingRect(points)
        return w * mm_per_px
    
    def _measure_lens_dimensions(
        self,
        masks: Dict[int, np.ndarray],
        mm_per_px: float
    ) -> Tuple[float, float]:
        """Measure lens width and height."""
        dimensions = []
        for class_id in (self.CLASS_LEFT_LENS, self.CLASS_RIGHT_LENS):
            lens_mask = masks.get(class_id)
            if lens_mask is None or not np.any(lens_mask):
                continue
            points = cv2.findNonZero((lens_mask > 0).astype(np.uint8))
            if points is None:
                continue
            x, y, width, height = cv2.boundingRect(points)
            dimensions.append((width * mm_per_px, height * mm_per_px))

        if not dimensions:
            # Legacy development fallback only. Production six-class extraction
            # rejects missing lens masks before reaching this method.
            frame_width = self._measure_frame_width(masks, mm_per_px)
            return frame_width * 0.38, frame_width * 0.35

        # Average both eyes when both are visible to reduce one-sided mask noise.
        return (
            float(np.mean([item[0] for item in dimensions])),
            float(np.mean([item[1] for item in dimensions])),
        )
    
    def _measure_bridge_width(self, masks: Dict[int, np.ndarray], mm_per_px: float) -> float:
        """Measure bridge width."""
        bridge_mask = masks.get(self.CLASS_BRIDGE)
        
        if bridge_mask is None or bridge_mask.sum() == 0:
            # Estimate: 12-15% of frame width
            frame_width = self._measure_frame_width(masks, mm_per_px)
            return frame_width * 0.13
        
        points = cv2.findNonZero((bridge_mask > 0).astype(np.uint8))
        if points is None:
            frame_width = self._measure_frame_width(masks, mm_per_px)
            return frame_width * 0.13
        x, y, w, h = cv2.boundingRect(points)
        return w * mm_per_px
    
    def _measure_temple_length(self, masks: Dict[int, np.ndarray], mm_per_px: float) -> float:
        """Measure temple length from contour arc length."""
        temple_mask = masks.get(self.CLASS_TEMPLE)
        
        if temple_mask is None or temple_mask.sum() == 0:
            # Standard temple length: ~140mm
            return 140.0
        
        contours, _ = cv2.findContours(temple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 140.0
        
        # Get longest contour (should be temple arm)
        longest = max(contours, key=lambda c: cv2.arcLength(c, False))
        arc_length_px = cv2.arcLength(longest, False)
        
        # Temple length is typically 85-90% of arc length (accounting for curve)
        temple_length = arc_length_px * mm_per_px * 0.87
        
        # Clamp to reasonable range
        return max(120.0, min(150.0, temple_length))
    
    def _measure_temple_curve(self, masks: Dict[int, np.ndarray]) -> float:
        """Estimate temple curve angle from temple mask."""
        temple_mask = masks.get(self.CLASS_TEMPLE)
        
        if temple_mask is None or temple_mask.sum() == 0:
            return 28.0  # Default curve angle
        
        contours, _ = cv2.findContours(temple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 28.0
        
        longest = max(contours, key=lambda c: cv2.arcLength(c, False))
        
        if len(longest) < 3:
            return 28.0
        
        # Fit line to contour and measure angle
        pts = longest.reshape(-1, 2).astype(np.float32)
        
        # Get start and end points
        start = pts[0]
        end = pts[-1]
        
        # Calculate angle
        vec = end - start
        angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
        
        # Clamp to reasonable range (10-45 degrees)
        return max(10.0, min(45.0, angle))
    
    def _extract_lens_contours(
        self,
        masks: Dict[int, np.ndarray],
        image_h: int,
        image_w: int
    ) -> LensContour:
        """Extract normalized lens contours for 3D generation."""
        
        left_lens = self._extract_single_lens_contour(
            masks.get(self.CLASS_LEFT_LENS),
            image_h,
            image_w,
            "left"
        )
        
        right_lens = self._extract_single_lens_contour(
            masks.get(self.CLASS_RIGHT_LENS),
            image_h,
            image_w,
            "right"
        )
        
        return LensContour(left=left_lens, right=right_lens)
    
    def _extract_single_lens_contour(
        self,
        mask: Optional[np.ndarray],
        image_h: int,
        image_w: int,
        side: str
    ) -> List[List[float]]:
        """Extract and normalize a single lens contour."""
        
        if mask is None or mask.sum() == 0:
            return self._default_lens_polygon(side)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._default_lens_polygon(side)
        
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Normalize coordinates to [0, 1]
        points = []
        for pt in approx.reshape(-1, 2):
            x_norm = pt[0] / max(image_w, 1)
            y_norm = pt[1] / max(image_h, 1)
            points.append([round(x_norm, 4), round(y_norm, 4)])
        
        return points if len(points) >= 4 else self._default_lens_polygon(side)
    
    def _default_lens_polygon(self, side: str) -> List[List[float]]:
        """Default lens shape if detection fails."""
        if side == "left":
            return [
                [0.05, 0.2], [0.2, 0.05], [0.4, 0.05], [0.45, 0.3],
                [0.45, 0.7], [0.4, 0.95], [0.2, 0.95], [0.05, 0.8],
            ]
        return [
            [0.55, 0.05], [0.8, 0.05], [0.95, 0.2], [0.95, 0.8],
            [0.8, 0.95], [0.55, 0.95], [0.5, 0.7], [0.5, 0.3],
        ]

# Made with Bob
