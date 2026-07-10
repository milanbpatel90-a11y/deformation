"""
Synthetic Test Image Generator
Creates realistic glasses on faces for tuning UI validation.
No real images needed—generates synthetic ones instantly.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_face_shape(height: int, width: int) -> np.ndarray:
    """Create a simple face outline mask"""
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Face outline (ellipse)
    center = (width // 2, height // 2)
    axes = (width // 3, height // 2)
    cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
    
    # Eyes (two circles)
    eye_y = int(height * 0.35)
    left_eye = (width // 3, eye_y)
    right_eye = (2 * width // 3, eye_y)
    cv2.circle(img, left_eye, 15, 200, -1)
    cv2.circle(img, right_eye, 15, 200, -1)
    
    return img


def create_glasses_on_face(
    height: int = 480,
    width: int = 640,
    style: str = "metal",
    rotation: float = 0.0,
    position_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic image: face + glasses overlay
    
    Args:
        height, width: Image dimensions
        style: "metal", "plastic", "oversized", "cat_eye"
        rotation: Tilt angle (degrees)
        position_offset: (x, y) offset from center
    
    Returns:
        (rgb_image, glasses_mask)
    """
    
    # Create face background
    face = create_face_shape(height, width)
    
    # Blend into realistic face (add skin tone)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    skin_color = np.array([210, 180, 140])  # Beige skin tone
    
    for c in range(3):
        rgb[:, :, c] = np.where(face > 0, skin_color[c], 200)
    
    # Create glasses mask
    glasses_mask = np.zeros((height, width), dtype=np.uint8)
    
    center_x = width // 2 + position_offset[0]
    center_y = height // 2 - height // 6 + position_offset[1]
    
    if style == "metal":
        # Thin metal frames: rectangular with slight curve
        left_lens_center = (center_x - 70, center_y)
        right_lens_center = (center_x + 70, center_y)
        lens_width, lens_height = 60, 40
        
        # Draw left lens
        cv2.ellipse(glasses_mask, left_lens_center, (lens_width, lens_height), 
                    int(rotation), 0, 360, 255, -1)
        # Draw right lens
        cv2.ellipse(glasses_mask, right_lens_center, (lens_width, lens_height), 
                    int(rotation), 0, 360, 255, -1)
        # Bridge
        cv2.line(glasses_mask, 
                (left_lens_center[0] + lens_width, left_lens_center[1]),
                (right_lens_center[0] - lens_width, right_lens_center[1]),
                255, 8)
    
    elif style == "plastic":
        # Thicker plastic frames: rounder
        left_lens_center = (center_x - 65, center_y)
        right_lens_center = (center_x + 65, center_y)
        lens_width, lens_height = 55, 45
        
        cv2.ellipse(glasses_mask, left_lens_center, (lens_width, lens_height),
                    int(rotation), 0, 360, 255, -1)
        cv2.ellipse(glasses_mask, right_lens_center, (lens_width, lens_height),
                    int(rotation), 0, 360, 255, -1)
        cv2.line(glasses_mask,
                (left_lens_center[0] + lens_width, left_lens_center[1]),
                (right_lens_center[0] - lens_width, right_lens_center[1]),
                255, 10)
    
    elif style == "oversized":
        # Large frames
        left_lens_center = (center_x - 75, center_y)
        right_lens_center = (center_x + 75, center_y)
        lens_width, lens_height = 70, 55
        
        cv2.ellipse(glasses_mask, left_lens_center, (lens_width, lens_height),
                    int(rotation), 0, 360, 255, -1)
        cv2.ellipse(glasses_mask, right_lens_center, (lens_width, lens_height),
                    int(rotation), 0, 360, 255, -1)
        cv2.line(glasses_mask,
                (left_lens_center[0] + lens_width, left_lens_center[1]),
                (right_lens_center[0] - lens_width, right_lens_center[1]),
                255, 6)
    
    elif style == "cat_eye":
        # Cat-eye frames: angled rectangles
        left_lens_center = (center_x - 65, center_y)
        right_lens_center = (center_x + 65, center_y)
        
        # Left lens (tilted up on outer edge)
        pts_left = np.array([
            [left_lens_center[0] - 55, left_lens_center[1] + 35],
            [left_lens_center[0] - 55, left_lens_center[1] - 35],
            [left_lens_center[0] + 55, left_lens_center[1] - 25],
            [left_lens_center[0] + 55, left_lens_center[1] + 45],
        ], dtype=np.int32)
        cv2.fillPoly(glasses_mask, [pts_left], 255)
        
        # Right lens (tilted up on inner edge)
        pts_right = np.array([
            [right_lens_center[0] - 55, right_lens_center[1] - 25],
            [right_lens_center[0] - 55, right_lens_center[1] + 45],
            [right_lens_center[0] + 55, right_lens_center[1] + 35],
            [right_lens_center[0] + 55, right_lens_center[1] - 35],
        ], dtype=np.int32)
        cv2.fillPoly(glasses_mask, [pts_right], 255)
    
    # Blend glasses into RGB image
    rgb_with_glasses = rgb.copy()
    glasses_color = np.array([50, 50, 50])  # Dark gray
    
    for c in range(3):
        rgb_with_glasses[:, :, c] = np.where(
            glasses_mask > 0,
            glasses_color[c],
            rgb_with_glasses[:, :, c]
        )
    
    return rgb_with_glasses, glasses_mask


def generate_test_set(output_dir: str = "test_images", count: int = 10):
    """Generate a set of test images for tuning UI validation"""
    
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    styles = ["metal", "plastic", "oversized", "cat_eye"]
    
    logger.info(f"Generating {count} test images in {output_dir}/")
    
    for i in range(count):
        style = styles[i % len(styles)]
        rotation = np.random.uniform(-15, 15)
        offset_x = np.random.randint(-20, 20)
        offset_y = np.random.randint(-10, 10)
        
        rgb, mask = create_glasses_on_face(
            height=480,
            width=640,
            style=style,
            rotation=rotation,
            position_offset=(offset_x, offset_y),
        )
        
        # Save RGB image
        img_path = out_path / f"test_{i:03d}_{style}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Save mask
        mask_path = out_path / f"test_{i:03d}_{style}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        logger.info(f"  [{i+1}/{count}] {img_path.name}")
    
    logger.info(f"\n✓ Generated {count} test images")
    logger.info(f"  Ready to upload to tuning UI: http://localhost:8001")
    logger.info(f"  Or use for training dataset in dataset/images/")


def create_single_test_image(
    output_path: str = "test_single.jpg",
    style: str = "metal",
) -> str:
    """Create a single high-quality test image"""
    
    rgb, mask = create_glasses_on_face(
        height=720,
        width=960,
        style=style,
        rotation=5.0,
    )
    
    cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    logger.info(f"✓ Created test image: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic test images")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output", default="test_images", help="Output directory")
    parser.add_argument("--single", action="store_true", help="Generate one high-quality image")
    parser.add_argument("--style", default="metal", help="Frame style for single image")
    
    args = parser.parse_args()
    
    if args.single:
        create_single_test_image(f"test_single_{args.style}.jpg", style=args.style)
    else:
        generate_test_set(args.output, args.count)
