"""View classifier to identify camera angles (front, side, top, perspective) of glasses."""

from __future__ import annotations
import cv2
import numpy as np


class ViewClassifier:
    """Classify eyewear images/masks into Front, Side, Top, or Perspective views."""

    def classify_view(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify the view angle based on image dimensions and mask shape properties.
        
        Returns: 'front', 'side', 'top', or 'perspective'
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "front"  # default fallback
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w == 0 or h == 0:
            return "front"
            
        aspect_ratio = float(w) / h
        
        # Calculate horizontal symmetry
        mid_x = x + w // 2
        left_half = mask[y:y+h, x:mid_x]
        right_half = mask[y:y+h, mid_x:x+w]
        
        left_area = float(np.count_nonzero(left_half))
        right_area = float(np.count_nonzero(right_half))
        
        max_area = max(left_area, right_area, 1.0)
        symmetry = min(left_area, right_area) / max_area
        
        # Classify based on aspect ratio, size, and symmetry
        # Fixed thresholds (2024-11-10) to improve classification accuracy
        # - Side view: more lenient (2.5 instead of 4.0)
        # - Top view: much stricter (0.95 instead of 1.3)
        # - Front view: explicit detection range
        # - Perspective: based on asymmetry
        
        if aspect_ratio >= 2.5:
            return "side"
        elif aspect_ratio < 0.95:
            return "top"
        elif 1.5 <= aspect_ratio < 2.5 and symmetry >= 0.72:
            return "front"
        elif symmetry < 0.68:
            return "perspective"
        else:
            return "front"  # safe default
