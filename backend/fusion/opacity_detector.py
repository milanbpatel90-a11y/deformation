"""Opacity detector to determine lens transparency and dominant color for sunglasses."""

from __future__ import annotations
import cv2
import numpy as np
from backend.models import LensContour


class OpacityDetector:
    """Detects opacity and dominant color of lens regions to identify sunglasses."""

    def detect(self, image: np.ndarray, lens_contour: LensContour) -> tuple[float, str]:
        """
        Detect lens opacity and dominant lens color.
        
        Returns:
            Tuple of (opacity: float, color_hex: str)
            Where opacity is 0.0 (completely clear) to 1.0 (opaque/black).
        """
        h, w = image.shape[:2]
        
        # Helper to convert normalized polygon to pixel contour
        def to_pixel_pts(polygon: list[list[float]]) -> np.ndarray | None:
            if not polygon or len(polygon) < 3:
                return None
            pts = []
            for p in polygon:
                px = int(np.clip(p[0] * w, 0, w - 1))
                py = int(np.clip(p[1] * h, 0, h - 1))
                pts.append([px, py])
            return np.array(pts, dtype=np.int32)

        left_pts = to_pixel_pts(lens_contour.left)
        right_pts = to_pixel_pts(lens_contour.right)
        
        # If no contours are available, fallback to defaults (clear lens)
        if left_pts is None or right_pts is None:
            return 0.31, "#ffffff"

        # Create a mask specifically for the lens regions
        lens_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lens_mask, [left_pts], -1, 255, -1)
        cv2.drawContours(lens_mask, [right_pts], -1, 255, -1)
        
        # Calculate average color and brightness of lens region
        lens_pixels = image[lens_mask == 255]
        if len(lens_pixels) == 0:
            return 0.31, "#ffffff"
            
        # Average color in BGR
        avg_bgr = np.mean(lens_pixels, axis=0)
        avg_b, avg_g, avg_r = avg_bgr[0], avg_bgr[1], avg_bgr[2]
        
        # Dominant color hex
        color_hex = f"#{int(avg_r):02x}{int(avg_g):02x}{int(avg_b):02x}"
        
        # Brightness (Value in HSV or Grayscale equivalent)
        brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        
        # Calculate opacity based on brightness:
        # A dark lens has high opacity. A bright/white lens has low opacity.
        opacity = 0.85 - (brightness / 255.0) * (0.85 - 0.31)
        
        # Keep opacity in safe boundaries
        opacity = float(np.clip(opacity, 0.25, 0.9))
        
        return opacity, color_hex
