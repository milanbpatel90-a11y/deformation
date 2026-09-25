from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.materials.pbr import lens_material
from backend.models import Measurements
from backend.pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.template_matching.template_matcher import TemplateMatcher
from deformation_tuning_api import _agreement


def _view(view_type: str, **measurements):
    base = {
        "frame_width": 140.0,
        "lens_width": 52.0,
        "lens_height": 42.0,
        "bridge_width": 18.0,
        "temple_length": 140.0,
    }
    base.update(measurements)
    return {"view_type": view_type, "measurements": base}


def test_agreement_marks_single_relevant_sample_insufficient():
    agreement, warnings = _agreement([
        _view("front"),
        _view("side"),
        _view("top"),
        _view("top"),
    ])

    assert agreement["frame_width"]["status"] == "insufficient_samples"
    assert agreement["frame_width"]["std_dev"] is None
    assert agreement["frame_width"]["flag"] is None
    assert agreement["frame_width"]["sample_count"] == 1

    assert agreement["temple_length"]["status"] == "insufficient_samples"
    assert agreement["temple_length"]["sample_count"] == 1
    assert any("insufficient relevant views" in warning for warning in warnings)


def test_agreement_reports_evidence_when_multiple_relevant_views_exist():
    agreement, warnings = _agreement([
        _view("front", frame_width=140.0),
        _view("perspective", frame_width=141.0),
        _view("side", temple_length=138.0),
        _view("side", temple_length=141.0),
    ])

    frame = agreement["frame_width"]
    temple = agreement["temple_length"]

    assert frame["sample_count"] == 2
    assert frame["status"] == "agreement"
    assert frame["flag"] is False
    assert frame["std_dev"] is not None

    assert temple["sample_count"] == 2
    assert temple["status"] == "agreement"
    assert temple["flag"] is False
    assert not any("frame_width varies" in warning for warning in warnings)


def test_agreement_flags_large_disagreement():
    agreement, warnings = _agreement([
        _view("front", bridge_width=14.0),
        _view("perspective", bridge_width=24.0),
        _view("side"),
        _view("top"),
    ])

    bridge = agreement["bridge_width"]
    assert bridge["status"] == "disagreement"
    assert bridge["flag"] is True
    assert bridge["sample_count"] == 2
    assert bridge["std_dev"] == 5.0
    assert any("bridge_width varies" in warning for warning in warnings)


def test_dashboard_has_production_input_guards():
    dashboard = Path("toolkit/dashboard.html").read_text(encoding="utf-8")

    assert "const MIN_FILES = 4;" in dashboard
    assert "const MAX_FILES = 5;" in dashboard
    assert "state.busy" in dashboard
    assert "AbortController" in dashboard
    assert "/api/reconstruct/multi-view" in dashboard
    assert "sampleResponse" not in dashboard



def test_manual_measurements_have_production_bounds():
    valid = Measurements(
        frame_width=140,
        lens_width=52,
        lens_height=42,
        bridge_width=18,
        temple_length=140,
    )
    assert valid.frame_width == 140

    with pytest.raises(ValidationError):
        Measurements(
            frame_width=250,
            lens_width=52,
            lens_height=42,
            bridge_width=18,
            temple_length=140,
        )


def test_template_registry_dictionary_is_supported(tmp_path):
    (tmp_path / "registry.json").write_text(
        '{"rectangle_plastic": {"name": "rectangle_plastic"}}',
        encoding="utf-8",
    )
    matcher = TemplateMatcher(TemplateLibrary(tmp_path))
    assert matcher._registry_template_names() == ["rectangle_plastic"]


def test_shared_mesh_aliases_are_marked_unsafe():
    descriptor = SimpleNamespace(
        raw={
            "mesh_aliases": {
                "Frame": "front_mesh",
                "Bridge": "front_mesh",
                "LeftRim": "front_mesh",
                "RightRim": "front_mesh",
                "LeftTemple": "left_temple",
                "RightTemple": "right_temple",
            }
        }
    )
    warnings = DeformationPipeline._template_safety_warnings(descriptor)
    assert warnings
    assert "same mesh" in warnings[0]


def test_dashboard_requires_manual_measurements_and_supports_overrides():
    dashboard = Path("toolkit/dashboard.html").read_text(encoding="utf-8")

    for field_id in (
        "frameWidth",
        "lensWidth",
        "lensHeight",
        "bridgeWidth",
        "templeLength",
        "shapeOverride",
        "materialOverride",
    ):
        assert f'id="{field_id}"' in dashboard

    assert "form.append('shape'" in dashboard
    assert "form.append('material'" in dashboard
    assert "Manual input" in dashboard



def test_lens_material_exports_as_transparent_blend():
    measurements = Measurements(
        frame_width=140,
        lens_width=52,
        lens_height=42,
        bridge_width=18,
        temple_length=140,
        lens_opacity=0.4,
    )
    material = lens_material(measurements)
    assert material.alphaMode == "BLEND"
    assert material.doubleSided is True


def test_viewer_scales_against_model_width_not_raw_face_width():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert "currentModelBaseWidth" in viewer
    assert "target_width / currentModelBaseWidth" in viewer
    assert "face_width * 0.72" not in viewer


def test_main_api_marks_legacy_single_view_and_requires_manual_multiview_inputs():
    api = Path("backend/api/main.py").read_text(encoding="utf-8")
    assert '@app.post("/api/deform", deprecated=True)' in api
    assert "frame_width: float = Form(...)" in api
    assert "manual_measurements" in api
    assert "traceback.format_exc" not in api
