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
            return "front"
            
        aspect_ratio = float(w) / h
        
        # Calculate horizontal symmetry
        mid_x = x + w // 2
        left_half = mask[y:y+h, x:mid_x]
        right_half = mask[y:y+h, mid_x:x+w]
        
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
        bilateral_similarity = (2.0 * overlap / max(np.count_nonzero(left_compare) +
                                                     np.count_nonzero(right_compare), 1.0))
        symmetry = min(area_balance, bilateral_similarity)
        
        # Classify based on aspect ratio, size, and symmetry
        if aspect_ratio >= 4.0:
            return "side"
        elif aspect_ratio < 1.0:
            return "top"
        elif symmetry < 0.30 or aspect_ratio < 1.45:
            return "perspective"
        else:
            return "front"
