import numpy as np
from PIL import Image

from glasses3d.pipeline.classify import classify_frame
from glasses3d.pipeline.landmarks import detect_landmarks
from glasses3d.pipeline.measurements import extract_measurements
from glasses3d.pipeline.preprocess import PARTS, preprocess


def test_preprocess_and_landmarks(tmp_path):
    image = np.full((512, 768, 4), 255, dtype=np.uint8)
    image[190:320, 120:650, :3] = 20
    path = tmp_path / "front.png"
    Image.fromarray(image).save(path)
    result = preprocess(path)
    assert set(result["front_masks"]) == set(PARTS)
    landmarks = detect_landmarks(result["front_masks"])
    measurements = extract_measurements(landmarks, 140)
    assert measurements.total_frame_width_mm == 140
    assert classify_frame(result["front_masks"])["frame_type"] in {"round", "rectangle", "aviator", "wayfarer"}


def test_side_fallback_is_stable():
    measurements = extract_measurements({"front": {"hinge_L": [0.2, 0.5], "hinge_R": [0.8, 0.5], "lens_outer_L": [0.2, 0.5], "lens_inner_L": [0.45, 0.5], "lens_inner_R": [0.55, 0.5], "frame_top": [0.5, 0.3], "frame_bottom": [0.5, 0.7]}}, 140)
    assert measurements.temple_length_mm == 135
