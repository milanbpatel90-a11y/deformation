import unittest
import numpy as np
from backend.models import Measurements, FrameMaterial, FrameShape, LensContour
from backend.fusion.view_classifier import ViewClassifier
from backend.fusion.measurement_fuser import MeasurementFuser
from backend.fusion.opacity_detector import OpacityDetector


class TestFusionModules(unittest.TestCase):
    def test_view_classifier_side_and_front(self) -> None:
        classifier = ViewClassifier()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 1. Front view mask (highly symmetric and moderately wide)
        front_mask = np.zeros((100, 100), dtype=np.uint8)
        front_mask[40:60, 20:80] = 255  # aspect ratio = 60 / 20 = 3.0, perfectly symmetric
        view_type = classifier.classify_view(dummy_img, front_mask)
        self.assertEqual(view_type, "front")

        # 2. Side view mask (very wide aspect ratio >= 4)
        side_mask = np.zeros((100, 100), dtype=np.uint8)
        side_mask[45:55, 10:90] = 255  # aspect ratio = 80 / 10 = 8.0
        view_type = classifier.classify_view(dummy_img, side_mask)
        self.assertEqual(view_type, "side")

    def test_measurement_fuser(self) -> None:
        fuser = MeasurementFuser()
        
        front_ms = Measurements(
            frame_width=140.0,
            lens_width=52.0,
            lens_height=48.0,
            bridge_width=18.0,
            temple_length=135.0,
            rim_thickness=1.2,
            material=FrameMaterial.METAL,
            shape=FrameShape.GEOMETRIC,
            nose_pads=True,
            color="#d9a7a2",
        )
        
        side_ms = Measurements(
            frame_width=135.0,
            lens_width=50.0,
            lens_height=46.0,
            bridge_width=16.0,
            temple_length=145.0,
            rim_thickness=1.2,
            material=FrameMaterial.METAL,
            shape=FrameShape.GEOMETRIC,
            nose_pads=True,
            temple_curve_angle=32.0,
            color="#d9a7a2",
        )
        
        top_ms = Measurements(
            frame_width=135.0,
            lens_width=50.0,
            lens_height=46.0,
            bridge_width=16.0,
            temple_length=135.0,
            rim_thickness=2.45,
            material=FrameMaterial.METAL,
            shape=FrameShape.GEOMETRIC,
            nose_pads=True,
            color="#d9a7a2",
        )
        
        # Fuse front + side + top measurements
        fused = fuser.fuse([
            ("front", front_ms),
            ("side", side_ms),
            ("top", top_ms),
        ])
        
        self.assertEqual(fused.frame_width, 140.0)  # prefers front
        self.assertEqual(fused.lens_width, 52.0)    # prefers front
        self.assertEqual(fused.temple_length, 145.0)  # prefers side
        self.assertEqual(fused.temple_curve_angle, 32.0)  # prefers side
        self.assertEqual(fused.rim_thickness, 2.45)  # prefers top

    def test_opacity_detector(self) -> None:
        detector = OpacityDetector()
        
        # Create a bright white/gray image (high brightness -> low opacity)
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        
        # Create a dark image (low brightness -> high opacity)
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 20
        
        contour = LensContour(
            left=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.9], [0.1, 0.9]],
            right=[[0.6, 0.1], [0.9, 0.1], [0.9, 0.9], [0.6, 0.9]],
        )
        
        opacity_bright, color_bright = detector.detect(bright_img, contour)
        opacity_dark, color_dark = detector.detect(dark_img, contour)
        
        self.assertLess(opacity_bright, 0.4)
        self.assertGreater(opacity_dark, 0.7)
        self.assertEqual(color_dark.lower(), "#141414")


if __name__ == "__main__":
    unittest.main()
