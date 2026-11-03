from __future__ import annotations

import numpy as np
import pytest

from backend.segmentation.segmenter import GlassesSegmenter, _validate_model_names


class FakeBoxes:
    def __init__(self, classes, confidences):
        self.cls = np.asarray(classes, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)


class FakeMasks:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class FakeResult:
    def __init__(self, masks, classes, confidences):
        self.masks = FakeMasks(masks)
        self.boxes = FakeBoxes(classes, confidences)


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def __call__(self, image, **kwargs):
        self.kwargs = kwargs
        return [self.result]


def _segmenter(model, names, confidence=0.25):
    segmenter = object.__new__(GlassesSegmenter)
    segmenter._model = model
    segmenter.model_type = "yolo:test.pt"
    segmenter.model_names = names
    segmenter.model_class_count = len(names)
    segmenter.supports_part_masks = len(names) == 6
    segmenter.confidence_threshold = confidence
    return segmenter


def test_yolo_keeps_each_part_mask_separate_and_returns_union():
    rim = np.zeros((8, 12), dtype=np.float32)
    rim[1:4, 1:5] = 1
    temple = np.zeros((8, 12), dtype=np.float32)
    temple[5:7, 7:11] = 1
    names = {0: "eyewear_rim", 1: "eyewear_temple"}
    # This test exercises the generic inference pairing logic with a small
    # two-class stub; production loading still only accepts 1 or 6 classes.
    segmenter = _segmenter(
        FakeModel(FakeResult([rim, temple], [0, 1], [0.95, 0.90])),
        names,
    )

    result = segmenter._segment_yolo(np.zeros((8, 12, 3), dtype=np.uint8))

    assert set(result["part_masks"]) == {"eyewear_rim", "eyewear_temple"}
    assert np.count_nonzero(result["part_masks"]["eyewear_rim"]) == 12
    assert np.count_nonzero(result["part_masks"]["eyewear_temple"]) == 8
    assert np.count_nonzero(result["front"]) == 20
    assert np.all(result["part_masks"]["eyewear_rim"][5:7, 7:11] == 0)
    assert result["observed_views"] == ["front"]


def test_yolo_filters_low_confidence_detections():
    visible = np.ones((8, 12), dtype=np.float32)
    segmenter = _segmenter(
        FakeModel(FakeResult([visible], [0], [0.10])),
        {0: "eyewear"},
        confidence=0.5,
    )

    result = segmenter._segment_yolo(np.zeros((8, 12, 3), dtype=np.uint8))

    assert result["detections"] == 0
    assert np.count_nonzero(result["front"]) == 0


def test_yolo_rejects_misaligned_masks_and_class_ids():
    mask = np.ones((8, 12), dtype=np.float32)
    segmenter = _segmenter(
        FakeModel(FakeResult([mask], [1, 0], [0.9, 0.9])),
        {0: "eyewear"},
    )
    with pytest.raises(ValueError, match="different number of masks"):
        segmenter._segment_yolo(np.zeros((8, 12, 3), dtype=np.uint8))


def test_yolo_rejects_unknown_class_ids():
    mask = np.ones((8, 12), dtype=np.float32)
    segmenter = _segmenter(
        FakeModel(FakeResult([mask], [2], [0.9])),
        {0: "eyewear", 1: "background"},
    )
    with pytest.raises(ValueError, match="unknown class ID"):
        segmenter._segment_yolo(np.zeros((8, 12, 3), dtype=np.uint8))


def test_segment_rejects_empty_or_wrong_shape_image():
    segmenter = _segmenter(FakeModel(None), {0: "eyewear"})
    with pytest.raises(ValueError, match="non-empty BGR image"):
        segmenter.segment(np.zeros((0, 12, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="non-empty BGR image"):
        segmenter.segment(np.zeros((8, 12), dtype=np.uint8))


def test_single_class_loader_rejects_unrelated_or_ambiguous_labels():
    with pytest.raises(ValueError, match="Unsupported one-class checkpoint label"):
        _validate_model_names({0: "person"})
    with pytest.raises(ValueError, match="Unsupported one-class checkpoint label"):
        _validate_model_names({0: "class0"})


def test_six_class_contract_accepts_temple_legs_alias():
    count, names, supports_parts = _validate_model_names({
        0: "eyewear_rim",
        1: "legs",
        2: "bridge",
        3: "left_lens",
        4: "right_lens",
        5: "nose_pad",
    })
    assert count == 6
    assert names[1] == "legs"
    assert supports_parts is True
