import cv2
import numpy as np

from backend.classifier.shape_classifier import ShapeClassifier
from backend.models import BridgeType, FrameFamily, FrameMaterial, FrameShape, RimType
from backend.template_library.loader import TemplateLibrary
from backend.template_matching.feature_extractor import EyewearFeatureSet
from backend.template_matching.template_matcher import TemplateMatcher


def test_wide_acetate_full_rim_is_not_misclassified_as_aviator():
    # Product images commonly show the whole frame as a wide silhouette. A
    # high-contrast acetate pattern reproduces the signal that used to trigger
    # the old width-only aviator heuristic.
    image = np.full((180, 420, 3), 245, dtype=np.uint8)
    rng = np.random.default_rng(7)
    image[45:125, 50:370] = rng.integers(0, 180, size=(80, 320, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (50, 45), (369, 124), 255, thickness=-1)

    style = ShapeClassifier().classify_style(image, mask)

    assert style.shape == FrameShape.RECTANGLE
    assert style.frame_family == FrameFamily.WAYFARER
    assert style.material.value == "acetate"
    assert style.metrics["wide_acetate_shape_correction"] is True


def test_wayfarer_style_prefers_matching_full_rim_template():
    features = EyewearFeatureSet(
        frame_family=FrameFamily.WAYFARER,
        rim_type=RimType.FULL_RIM,
        bridge_type=BridgeType.KEYHOLE,
        material=FrameMaterial.ACETATE,
        lens_aspect_ratio=1.25,
        frame_width=135,
        frame_height=47,
        temple_length=131,
        confidence=1,
    )

    match = TemplateMatcher(TemplateLibrary()).match(features)

    assert match.best.template.name == "RB_001"


def test_material_detection_uses_rim_pixels_not_lens_or_background():
    patterned_acetate = np.array([[0, 10, 20], [220, 180, 90]] * 20, dtype=np.uint8)
    matte_dark_frame = np.full((40, 3), 24, dtype=np.uint8)
    reflective_metal = np.full((40, 3), 175, dtype=np.uint8)

    classify = ShapeClassifier.classify_material_pixels
    assert classify(patterned_acetate).value == "acetate"
    assert classify(matte_dark_frame).value == "plastic"
    assert classify(reflective_metal).value == "metal"
