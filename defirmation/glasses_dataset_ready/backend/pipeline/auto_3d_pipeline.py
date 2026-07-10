"""
Automatic 3D Glasses Generation Pipeline
End-to-end pipeline: Image → Segmentation → Measurements → 3D GLB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np

from backend.models import Measurements, LensContour, FrameShape, FrameMaterial
from backend.segmentation.segmenter import GlassesSegmenter
from backend.measurement.mask_to_measurements import MaskToMeasurements
from backend.generator.parametric_glb_generator import ParametricGLBGenerator
from backend.classifier.shape_classifier import ShapeClassifier

logger = logging.getLogger(__name__)


class Auto3DPipeline:
    """
    Automatic 3D glasses generation pipeline.
    
    Workflow:
    1. Load product image
    2. Segment glasses parts (YOLO)
    3. Extract measurements from masks
    4. Generate parametric 3D model (GLB)
    """
    
    def __init__(
        self,
        yolo_model_path: Optional[str | Path] = None,
        reference_width_mm: float = 140.0
    ):
        """
        Initialize pipeline.
        
        Args:
            yolo_model_path: Path to trained YOLOv8-seg model
            reference_width_mm: Reference frame width for calibration
        """
        self.segmenter = GlassesSegmenter(yolo_model_path)
        self.mask_converter = MaskToMeasurements(reference_width_mm)
        self.glb_generator = ParametricGLBGenerator()
        self.shape_classifier = ShapeClassifier()
        
        logger.info("Auto3D Pipeline initialized")
        if yolo_model_path:
            logger.info(f"Using YOLO model: {yolo_model_path}")
        else:
            logger.warning("No YOLO model provided, using OpenCV fallback")
    
    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        material: Optional[FrameMaterial] = None,
        color: Optional[str] = None
    ) -> dict:
        """
        Process a single glasses image and generate 3D model.
        
        Args:
            image_path: Path to product image
            output_dir: Directory for output files
            material: Frame material (auto-detected if None)
            color: Frame color hex (auto-detected if None)
            
        Returns:
            Dict with paths and metadata
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing: {image_path.name}")
        
        # 1. Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 2. Segment glasses
        logger.info("Step 1/4: Segmenting glasses parts...")
        masks = self.segmenter.segment(image)
        
        # Save segmentation visualization
        seg_viz_path = output_dir / f"{image_path.stem}_segmentation.jpg"
        self._save_segmentation_viz(image, masks, seg_viz_path)
        
        # 3. Classify shape and material
        logger.info("Step 2/4: Classifying shape and material...")
        shape = self.shape_classifier.classify_shape(image, masks.get("front"))
        
        if material is None:
            material = self._detect_material(image, masks.get("front"))
        
        if color is None:
            color = self._detect_color(image, masks.get("front"))
        
        # 4. Extract measurements
        logger.info("Step 3/4: Extracting measurements...")
        
        # For YOLO results, we need to handle differently
        if hasattr(self.segmenter, '_model') and self.segmenter._model is not None:
            # Run YOLO again to get results object
            results = self.segmenter._model(image, verbose=False)
            measurements, lens_contour = self.mask_converter.extract_from_yolo_results(
                image, results, shape, material, color
            )
        else:
            # Fallback: use OpenCV masks
            measurements, lens_contour = self._extract_from_opencv_masks(
                image, masks, shape, material, color
            )
        
        # Save measurements
        measurements_path = output_dir / f"{image_path.stem}_measurements.json"
        with open(measurements_path, 'w') as f:
            json.dump(measurements.dict(), f, indent=2)
        
        # 5. Generate 3D model
        logger.info("Step 4/4: Generating 3D model...")
        glb_path = output_dir / f"{image_path.stem}.glb"
        self.glb_generator.generate_from_measurements(
            measurements, lens_contour, glb_path
        )
        
        logger.info(f"✓ Generated: {glb_path.name}")
        
        return {
            "input_image": str(image_path),
            "glb_model": str(glb_path),
            "measurements": str(measurements_path),
            "segmentation_viz": str(seg_viz_path),
            "measurements_data": measurements.dict(),
            "shape": shape.value,
            "material": material.value,
            "color": color
        }
    
    def process_batch(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.jpg"
    ) -> list[dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_dir: Directory containing product images
            output_dir: Directory for output files
            pattern: File pattern to match
            
        Returns:
            List of result dicts
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        image_files = sorted(image_dir.glob(pattern))
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")
            try:
                result = self.process_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                results.append({
                    "input_image": str(image_path),
                    "error": str(e)
                })
        
        # Save batch summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total": len(image_files),
                "successful": len([r for r in results if "error" not in r]),
                "failed": len([r for r in results if "error" in r]),
                "results": results
            }, f, indent=2)
        
        logger.info(f"\n✓ Batch complete. Summary: {summary_path}")
        return results
    
    def _extract_from_opencv_masks(
        self,
        image: np.ndarray,
        masks: dict,
        shape: FrameShape,
        material: FrameMaterial,
        color: str
    ) -> Tuple[Measurements, LensContour]:
        """Fallback: extract measurements from OpenCV masks."""
        from backend.measurement.extractor import MeasurementExtractor
        
        extractor = MeasurementExtractor()
        return extractor.extract_from_images(
            front=image,
            mask=masks.get("front"),
            shape=shape,
            material=material,
            color=color
        )
    
    def _detect_material(self, image: np.ndarray, mask: Optional[np.ndarray]) -> FrameMaterial:
        """Auto-detect frame material from image."""
        # Simple heuristic: check for metallic reflections
        if mask is None or mask.sum() == 0:
            return FrameMaterial.PLASTIC
        
        # Extract frame region
        masked = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Check for high-intensity specular highlights (metal)
        bright_pixels = (gray > 200).sum()
        total_pixels = (mask > 0).sum()
        
        if total_pixels > 0:
            brightness_ratio = bright_pixels / total_pixels
            if brightness_ratio > 0.15:
                return FrameMaterial.METAL
        
        return FrameMaterial.PLASTIC
    
    def _detect_color(self, image: np.ndarray, mask: Optional[np.ndarray]) -> str:
        """Auto-detect dominant frame color."""
        if mask is None or mask.sum() == 0:
            return "#000000"
        
        # Extract frame region
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Get mean color in masked region
        pixels = masked[mask > 0]
        if len(pixels) == 0:
            return "#000000"
        
        mean_color = pixels.mean(axis=0)
        b, g, r = mean_color
        
        # Convert to hex
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    
    def _save_segmentation_viz(
        self,
        image: np.ndarray,
        masks: dict,
        output_path: Path
    ) -> None:
        """Save visualization of segmentation results."""
        viz = image.copy()
        
        # Overlay masks with different colors
        colors = {
            "front": (0, 255, 0),    # Green
            "side": (255, 0, 0),     # Blue
            "full": (0, 0, 255)      # Red
        }
        
        for key, mask in masks.items():
            if mask is not None and mask.sum() > 0:
                color = colors.get(key, (255, 255, 255))
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                viz = cv2.addWeighted(viz, 0.7, colored_mask, 0.3, 0)
        
        cv2.imwrite(str(output_path), viz)


def main():
    """CLI interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automatic 3D Glasses Generation Pipeline"
    )
    parser.add_argument(
        "input",
        help="Input image or directory"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output_3d",
        help="Output directory"
    )
    parser.add_argument(
        "-m", "--model",
        help="Path to trained YOLO model (optional)"
    )
    parser.add_argument(
        "--material",
        choices=["metal", "plastic", "acetate", "titanium"],
        help="Frame material (auto-detected if not specified)"
    )
    parser.add_argument(
        "--color",
        help="Frame color in hex (e.g., #000000)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory in batch mode"
    )
    parser.add_argument(
        "--reference-width",
        type=float,
        default=140.0,
        help="Reference frame width in mm for calibration"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = Auto3DPipeline(
        yolo_model_path=args.model,
        reference_width_mm=args.reference_width
    )
    
    # Convert material string to enum
    material = None
    if args.material:
        material = FrameMaterial(args.material)
    
    # Process
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        pipeline.process_batch(input_path, args.output)
    else:
        pipeline.process_image(
            input_path,
            args.output,
            material=material,
            color=args.color
        )


if __name__ == "__main__":
    main()

# Made with Bob
