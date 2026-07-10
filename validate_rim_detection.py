"""
Rim Detection Pipeline Validation
Test the complete flow: image → segmentation → rim detection → rim_pull estimation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RimDetectionValidator:
    """Validate rim detection pipeline end-to-end"""
    
    def __init__(self):
        self.results = []
    
    def detect_rim_from_mask(
        self,
        mask: np.ndarray,
        min_area: int = 500,
    ) -> Tuple[List[int], float, float]:
        """
        Detect rim contour from glasses mask.
        
        Args:
            mask: Binary mask (0=background, 255=glasses)
            min_area: Minimum contour area to consider
        
        Returns:
            (rim_vertices, edge_confidence, fit_error)
        """
        
        # Apply morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in mask")
            return [], 0.0, 1.0
        
        # Find largest contour (glasses)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < min_area:
            logger.warning(f"Contour area {area} < min_area {min_area}")
            return [], 0.0, 1.0
        
        # Fit polygon to contour (8 vertices for rim)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.02 * perimeter
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract 8 rim vertices (if possible)
        if len(polygon) >= 8:
            # Resample to exactly 8 vertices
            rim_vertices = self._resample_polygon(polygon.reshape(-1, 2), 8)
        else:
            logger.warning(f"Polygon has {len(polygon)} vertices, need 8 for rim")
            rim_vertices = polygon.reshape(-1, 2)
        
        # Measure confidence
        edge_confidence = self._measure_edge_sharpness(mask_clean, largest_contour)
        
        # Measure fit error
        fit_error = self._measure_polygon_fit_error(largest_contour, polygon)
        
        # Convert to vertex indices (0-based list)
        vertex_ids = list(range(len(rim_vertices)))
        
        return vertex_ids, edge_confidence, fit_error
    
    def _resample_polygon(self, points: np.ndarray, n: int) -> np.ndarray:
        """Resample polygon to exactly n points"""
        
        # Calculate cumulative distances
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]
        
        # New sample points
        new_distances = np.linspace(0, total_length, n)
        
        # Interpolate
        resampled = np.array([
            np.interp(new_distances, cumulative, points[:, 0]),
            np.interp(new_distances, cumulative, points[:, 1]),
        ]).T
        
        return resampled.astype(np.int32)
    
    def _measure_edge_sharpness(self, mask: np.ndarray, contour: np.ndarray) -> float:
        """Measure edge detection confidence (0-1)"""
        
        # Apply Canny edge detection
        edges = cv2.Canny(mask, 50, 150)
        
        # Check how much of contour overlaps with edges
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, 2)
        
        overlap = cv2.bitwise_and(edges, contour_mask)
        overlap_ratio = np.count_nonzero(overlap) / (np.count_nonzero(contour_mask) + 1e-6)
        
        return min(1.0, overlap_ratio * 1.5)  # Scale to 0-1
    
    def _measure_polygon_fit_error(self, contour: np.ndarray, polygon: np.ndarray) -> float:
        """Measure how well polygon fits contour (0-1, lower is better)"""
        
        error = cv2.matchShapes(contour, polygon, cv2.CONTOURS_MATCH_I3, 0.0)
        
        # Normalize to 0-1 range
        normalized_error = min(1.0, error / 10.0)
        
        return normalized_error
    
    def estimate_rim_pull_strength(
        self,
        edge_confidence: float,
        fit_error: float,
    ) -> float:
        """
        Estimate rim_pull_strength from detection metrics.
        
        Heuristic:
        - High confidence + low error → high pull strength (aggressive deformation)
        - Low confidence + high error → low pull strength (conservative)
        
        Returns: 0.1-0.9 (clamped to sensible range)
        """
        
        strength = edge_confidence * (1.0 - fit_error)
        strength = np.clip(strength, 0.1, 0.9)
        
        return float(strength)
    
    def validate_image(self, image_path: str, style: str = "metal") -> Dict:
        """Validate a single image file"""
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return {}
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return {}
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold to create mask (for synthetic images)
        # In production, use YOLOv8-Seg
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect rim
        rim_vertices, edge_conf, fit_err = self.detect_rim_from_mask(mask)
        
        # Estimate pull strength
        rim_pull = self.estimate_rim_pull_strength(edge_conf, fit_err)
        
        result = {
            "image": str(image_path),
            "style": style,
            "detected_vertices": rim_vertices,
            "edge_confidence": float(edge_conf),
            "fit_error": float(fit_err),
            "estimated_rim_pull_strength": float(rim_pull),
        }
        
        self.results.append(result)
        
        logger.info(f"✓ {image_path.name}")
        logger.info(f"  Edge Confidence: {edge_conf:.1%}")
        logger.info(f"  Fit Error: {fit_err:.1%}")
        logger.info(f"  Estimated Rim Pull: {rim_pull:.2f}")
        
        return result
    
    def validate_directory(self, image_dir: str, pattern: str = "*.jpg") -> List[Dict]:
        """Validate all images in directory"""
        
        dir_path = Path(image_dir)
        images = list(dir_path.glob(pattern))
        
        if not images:
            logger.warning(f"No images matching {pattern} in {image_dir}")
            return []
        
        logger.info(f"Validating {len(images)} images from {image_dir}/\n")
        
        for img_path in sorted(images):
            self.validate_image(str(img_path))
        
        return self.results
    
    def summary(self) -> Dict:
        """Generate validation summary"""
        
        if not self.results:
            return {}
        
        edge_confs = [r["edge_confidence"] for r in self.results]
        fit_errs = [r["fit_error"] for r in self.results]
        rim_pulls = [r["estimated_rim_pull_strength"] for r in self.results]
        
        summary = {
            "total_images": len(self.results),
            "edge_confidence": {
                "mean": float(np.mean(edge_confs)),
                "min": float(np.min(edge_confs)),
                "max": float(np.max(edge_confs)),
            },
            "fit_error": {
                "mean": float(np.mean(fit_errs)),
                "min": float(np.min(fit_errs)),
                "max": float(np.max(fit_errs)),
            },
            "rim_pull_strength": {
                "mean": float(np.mean(rim_pulls)),
                "min": float(np.min(rim_pulls)),
                "max": float(np.max(rim_pulls)),
            },
        }
        
        return summary
    
    def save_report(self, output_path: str = "validation_report.json"):
        """Save validation results to JSON"""
        
        report = {
            "results": self.results,
            "summary": self.summary(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved: {output_path}")
        
        return report


def print_summary(validator: RimDetectionValidator):
    """Pretty-print validation summary"""
    
    summary = validator.summary()
    
    if not summary:
        logger.warning("No results to summarize")
        return
    
    print("\n" + "="*70)
    print("RIM DETECTION VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\nTotal Images: {summary['total_images']}")
    
    print(f"\nEdge Confidence (target: >0.85):")
    print(f"  Mean: {summary['edge_confidence']['mean']:.1%}")
    print(f"  Min:  {summary['edge_confidence']['min']:.1%}")
    print(f"  Max:  {summary['edge_confidence']['max']:.1%}")
    
    print(f"\nFit Error (target: <0.15):")
    print(f"  Mean: {summary['fit_error']['mean']:.1%}")
    print(f"  Min:  {summary['fit_error']['min']:.1%}")
    print(f"  Max:  {summary['fit_error']['max']:.1%}")
    
    print(f"\nEstimated Rim Pull Strength (0.1-0.9):")
    print(f"  Mean: {summary['rim_pull_strength']['mean']:.2f}")
    print(f"  Min:  {summary['rim_pull_strength']['min']:.2f}")
    print(f"  Max:  {summary['rim_pull_strength']['max']:.2f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate rim detection pipeline")
    parser.add_argument("--image", help="Single image to validate")
    parser.add_argument("--directory", help="Directory of images to validate")
    parser.add_argument("--pattern", default="*.jpg", help="Image filename pattern")
    parser.add_argument("--output", default="validation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    validator = RimDetectionValidator()
    
    if args.image:
        result = validator.validate_image(args.image)
        print(json.dumps(result, indent=2))
    
    elif args.directory:
        validator.validate_directory(args.directory, pattern=args.pattern)
        validator.save_report(args.output)
        print_summary(validator)
    
    else:
        # Default: validate test_images/
        if Path("test_images").exists():
            validator.validate_directory("test_images")
            validator.save_report(args.output)
            print_summary(validator)
        else:
            parser.print_help()
