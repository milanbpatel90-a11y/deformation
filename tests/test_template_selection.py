"""Auto-select must resolve to templates that really exist, with their own GLB and descriptor."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.classifier.shape_classifier import ShapeClassifier
from backend.models import FrameMaterial, FrameShape
from backend.template_library.loader import TemplateLibrary

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
NEW_TEMPLATES = ["cat_eye_acetate", "rectangle_acetate", "round_metal", "aviator_metal", "browline_semirimless"]


@pytest.fixture(scope="module")
def library() -> TemplateLibrary:
    return TemplateLibrary(TEMPLATES)


@pytest.mark.parametrize("name", NEW_TEMPLATES)
def test_new_template_is_discoverable_and_has_own_assets(name: str, library: TemplateLibrary) -> None:
    assert name in library.list_templates()
    assert (TEMPLATES / f"{name}.glb").is_file()
    assert (TEMPLATES / "descriptors" / f"{name}.json").is_file()
    info = library.load(name)
    assert Path(info.glb_path).name == f"{name}.glb"  # not the geometric_metal fallback


@pytest.mark.parametrize(
    "shape,material,expected",
    [
        (FrameShape.ROUND, FrameMaterial.METAL, "round_metal"),
        (FrameShape.AVIATOR, FrameMaterial.METAL, "aviator_metal"),
        (FrameShape.CAT_EYE, FrameMaterial.ACETATE, "cat_eye_acetate"),
        (FrameShape.BROWLINE, FrameMaterial.MIXED, "browline_semirimless"),
        (FrameShape.RECTANGLE, FrameMaterial.ACETATE, "rectangle_acetate"),
        (FrameShape.RECTANGLE, FrameMaterial.PLASTIC, "rectangle_plastic"),
        (FrameShape.GEOMETRIC, FrameMaterial.METAL, "geometric_metal"),
    ],
)
def test_shape_selects_matching_template(shape, material, expected, library: TemplateLibrary) -> None:
    name = ShapeClassifier().template_name_for(shape, material)
    assert name == expected
    assert name in library.list_templates()
