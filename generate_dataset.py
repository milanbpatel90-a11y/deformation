"""
Automated Dataset Generator
Generates synthetic images and perfectly matching YOLOv8-Seg annotations.
Completely bypasses manual CVAT annotation.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random

# Import the synthetic generator
from generate_test_images import create_glasses_on_face

def ensure_dirs():
    dirs = ['dataset/images/train', 'dataset/images/val', 
            'dataset/labels/train', 'dataset/labels/val',
            'dataset/masks']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def extract_yolo_polygons(mask: np.ndarray) -> list:
    """Extract normalized polygons from mask for YOLO format"""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
            
        # Simplify contour to reduce file size (YOLO doesn't need every single pixel)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Normalize coordinates to 0.0 - 1.0
        normalized = []
        for point in approx:
            x, y = point[0]
            normalized.append(f"{x/w:.6f} {y/h:.6f}")
            
        if len(normalized) >= 3:
            yolo_polygons.append("0 " + " ".join(normalized))
            
    return yolo_polygons

def generate_dataset(total_images=200, train_ratio=0.8):
    ensure_dirs()
    styles = ["metal", "plastic", "oversized", "cat_eye"]
    
    print(f"Generating {total_images} annotated synthetic images...")
    
    for i in range(total_images):
        split = "train" if random.random() < train_ratio else "val"
        style = random.choice(styles)
        rotation = np.random.uniform(-15, 15)
        offset_x = np.random.randint(-20, 20)
        offset_y = np.random.randint(-10, 10)
        
        rgb, mask = create_glasses_on_face(
            height=480, width=640,
            style=style, rotation=rotation,
            position_offset=(offset_x, offset_y)
        )
        
        # Paths
        base_name = f"synth_{i:04d}_{style}"
        img_path = f"dataset/images/{split}/{base_name}.jpg"
        label_path = f"dataset/labels/{split}/{base_name}.txt"
        mask_path = f"dataset/masks/{base_name}.png"
        
        # Save image and mask
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        
        # Generate and save YOLO label
        polygons = extract_yolo_polygons(mask)
        with open(label_path, "w") as f:
            f.write("\n".join(polygons))
            
        if (i+1) % 20 == 0:
            print(f"[{i+1}/{total_images}] Generated...")
            
    # Write dataset yaml
    yaml_content = f"""path: {Path('dataset').absolute()}
train: images/train
val: images/val

names:
  0: eyewear
"""
    with open("dataset/data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\nDone! Dataset generated in 'dataset/'")
    print(f"You can now train YOLO with: yolo task=segment mode=train data=dataset/data.yaml model=yolov8n-seg.pt epochs=20")

if __name__ == "__main__":
    generate_dataset(200)
