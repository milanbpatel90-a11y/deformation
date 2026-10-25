from __future__ import annotations

import pytest

from backend.segmentation.segmenter import _find_model, _validate_model_names


def test_accepts_single_class_eyewear_model():
    count, names, supports_parts = _validate_model_names({0: "eyewear"})
    assert count == 1
    assert names[0] == "eyewear"
    assert supports_parts is False


def test_accepts_the_six_class_measurement_contract():
    names = {
        0: "eyewear_rim",
        1: "eyewear_temple",
        2: "bridge",
        3: "left_lens",
        4: "right_lens",
        5: "nose_pad",
    }
    count, _, supports_parts = _validate_model_names(names)
    assert count == 6
    assert supports_parts is True


def test_rejects_class_count_that_does_not_match_contract():
    with pytest.raises(ValueError, match="expected 1 class.*or 6"):
        _validate_model_names({0: "eyewear", 1: "background"})


def test_rejects_wrong_multi_class_order():
    names = {
        0: "eyewear",
        1: "temple",
        2: "bridge",
        3: "left_lens",
        4: "right_lens",
        5: "nose_pad",
    }
    with pytest.raises(ValueError, match="Wrong class contract at ID 0"):
        _validate_model_names(names)


def test_checkpoint_discovery_prefers_fine_tuned_weights_and_ignores_generic(tmp_path, monkeypatch):
    import backend.segmentation.segmenter as module

    monkeypatch.setattr(module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DEFIRM_YOLO_MODEL", raising=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fine_tuned = model_dir / "best.pt"
    fine_tuned.write_bytes(b"checkpoint-placeholder")
    (tmp_path / "yolov8n-seg.pt").write_bytes(b"generic-placeholder")
    assert _find_model() == fine_tuned.resolve()


def test_configured_missing_checkpoint_fails_instead_of_silent_fallback(tmp_path, monkeypatch):
    import backend.segmentation.segmenter as module

    monkeypatch.setattr(module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DEFIRM_YOLO_MODEL", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="missing file"):
        _find_model()
