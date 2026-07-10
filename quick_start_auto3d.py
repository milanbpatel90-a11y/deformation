"""
Quick Start Script for Automatic 3D Glasses Generation
Run this to test the complete pipeline with a single command.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed."""
    logger.info("Checking dependencies...")
    
    required = [
        'ultralytics',
        'trimesh',
        'pygltflib',
        'shapely',
        'cv2',
        'numpy'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.error(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        logger.error("\nMissing packages. Install with:")
        logger.error("pip install ultralytics trimesh pygltflib shapely opencv-python numpy")
        return False
    
    logger.info("✓ All dependencies installed\n")
    return True


def setup_directories():
    """Create necessary directories."""
    logger.info("Setting up directories...")
    
    dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val',
        'output_3d',
        'runs/segment'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {dir_path}")
    
    logger.info("✓ Directories ready\n")


def verify_yolo():
    """Verify YOLO installation."""
    logger.info("Verifying YOLO installation...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a model
        model = YOLO('yolov8n-seg.pt')
        logger.info("  ✓ YOLO model loaded successfully")
        logger.info(f"  ✓ Model: {model.model_name if hasattr(model, 'model_name') else 'yolov8n-seg'}")
        return True
    except Exception as e:
        logger.error(f"  ✗ YOLO verification failed: {e}")
        return False


def test_pipeline():
    """Test the pipeline with test images."""
    logger.info("\nTesting pipeline with test images...")
    
    # Check if test images exist
    test_dir = Path('test_images')
    if not test_dir.exists() or not list(test_dir.glob('*.jpg')):
        logger.warning("No test images found in test_images/")
        logger.info("You can add test images to test_images/ directory")
        return
    
    try:
        from backend.pipeline.auto_3d_pipeline import Auto3DPipeline
        
        # Initialize pipeline (without trained model for now)
        pipeline = Auto3DPipeline()
        
        # Process first test image
        test_images = list(test_dir.glob('*.jpg'))
        if test_images:
            test_image = test_images[0]
            logger.info(f"Processing test image: {test_image.name}")
            
            result = pipeline.process_image(
                test_image,
                'output_3d'
            )
            
            logger.info(f"✓ Generated: {result['glb_model']}")
            logger.info(f"✓ Measurements: {result['measurements']}")
            logger.info(f"✓ Segmentation viz: {result['segmentation_viz']}")
            
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        logger.info("This is expected if you haven't trained a model yet")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNEXT STEPS:\n")
    
    print("1. PREPARE DATASET")
    print("   - Add images to: dataset/images/train/")
    print("   - Annotate using CVAT (see CVAT_SETUP_AND_ANNOTATION.md)")
    print("   - Export labels to: dataset/labels/train/")
    print()
    
    print("2. TRAIN YOLO MODEL")
    print("   python train_yolov8_seg.py train --dataset-yaml data.yaml --epochs 50")
    print()
    
    print("3. GENERATE 3D MODELS")
    print("   # Single image")
    print("   python backend/pipeline/auto_3d_pipeline.py glasses.jpg -o output -m runs/segment/train/weights/best.pt")
    print()
    print("   # Batch processing")
    print("   python backend/pipeline/auto_3d_pipeline.py ./images/ -o output -m best.pt --batch")
    print()
    
    print("4. VIEW RESULTS")
    print("   - GLB models: output_3d/*.glb")
    print("   - Measurements: output_3d/*_measurements.json")
    print("   - Visualizations: output_3d/*_segmentation.jpg")
    print()
    
    print("DOCUMENTATION")
    print("   - Complete guide: AUTO_3D_PIPELINE_GUIDE.md")
    print("   - CVAT setup: CVAT_SETUP_AND_ANNOTATION.md")
    print("   - Architecture: ARCHITECTURE.md")
    print()
    
    print("QUICK TEST (without training)")
    print("   python quick_start_auto3d.py --test")
    print()
    
    print("="*60)


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("AUTOMATIC 3D GLASSES GENERATION - QUICK START")
    print("="*60 + "\n")
    
    # Check for test flag
    if '--test' in sys.argv:
        test_pipeline()
        return
    
    # Run setup steps
    if not check_dependencies():
        logger.error("\n❌ Setup failed: Missing dependencies")
        logger.error("Install with: pip install -r requirements.txt")
        return
    
    setup_directories()
    verify_yolo()
    
    # Print next steps
    print_next_steps()
    
    logger.info("✅ Setup complete! Follow the next steps above.")


if __name__ == "__main__":
    main()

# Made with Bob
