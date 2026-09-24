from pathlib import Path

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
