import cv2
import numpy as np
from backend.api.main import pipeline
from backend.materials.appearance import apply_image_appearance
from backend.models import Measurements


def test_image_colour_replaces_stale_picker_without_changing_sizes():
    image = np.full((180, 360, 3), 255, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (30, 40), (330, 140), 255, 8)
    image[mask > 0] = (40, 80, 120)
    manual = Measurements(frame_width=142, lens_width=56, lens_height=37,
                          bridge_width=18, temple_length=130, rim_thickness=1.2,
                          color="#d9a7a2")
    detected, style, _ = apply_image_appearance(pipeline, manual, image, mask)
    assert detected.color == "#785028"
    assert detected.shape == style.shape
    assert detected.material == style.material
    for field in ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length", "rim_thickness"):
        assert getattr(detected, field) == getattr(manual, field)
    assert manual.color == "#d9a7a2"
