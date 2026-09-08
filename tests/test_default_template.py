from pathlib import Path

from backend.template_library.loader import DEFAULT_TEMPLATE_NAME, TemplateLibrary
from backend.template_matching.template_matcher import TemplateMatcher


def test_gold_template_is_available_and_uses_runtime_geometry():
    library = TemplateLibrary()

    info = library.load(DEFAULT_TEMPLATE_NAME)

    assert info.name == DEFAULT_TEMPLATE_NAME
    assert Path(info.glb_path).name == "geometric_metal.glb"
    assert Path(info.glb_path).exists()


def test_matcher_uses_gold_template_when_no_override():
    library = TemplateLibrary()
    matcher = TemplateMatcher(library)

    # The matcher only needs a feature object to score the selected template.
    from backend.template_matching.feature_extractor import EyewearFeatureSet
    from backend.models import FrameFamily, FrameMaterial, RimType, BridgeType

    features = EyewearFeatureSet(
        frame_family=FrameFamily.RECTANGLE,
        material=FrameMaterial.METAL,
        rim_type=RimType.FULL_RIM,
        bridge_type=BridgeType.PAD,
        lens_aspect_ratio=1.2,
        frame_width=140.0,
        frame_height=56.0,
        temple_length=145.0,
        wrap_angle=None,
        confidence=1.0,
    )

    result = matcher.match(features)

    assert result.best.template.name == DEFAULT_TEMPLATE_NAME
