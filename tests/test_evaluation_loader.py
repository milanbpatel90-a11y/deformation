import json

import pytest

from backend.evaluation.dataset import GroundTruthDataset, DatasetValidationError


def valid_payload():
    fields = {"frame_width": 140, "bridge_width": 18, "lens_width": 50, "lens_height": 40, "temple_length": 145}
    return {
        "version": 1,
        "units": "mm",
        "samples": [{
            "id": "rb2140_001",
            "image": "images/rb2140_001.jpg",
            "archetype": "wayfarer",
            "source": "tape_measure",
            "measured_by": "mbp",
            "ground_truth": fields,
            "tolerance": {key: 1.0 for key in fields},
        }],
    }


def test_loader_accepts_canonical_mm_schema(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(valid_payload()), encoding="utf-8")
    dataset = GroundTruthDataset.load(path)
    assert dataset.units == "mm"
    assert dataset.samples[0].ground_truth["frame_width"] == 140.0


def test_loader_rejects_non_mm_units(tmp_path):
    payload = valid_payload()
    payload["units"] = "cm"
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="mm"):
        GroundTruthDataset.load(path)


def test_loader_rejects_unknown_fields(tmp_path):
    payload = valid_payload()
    payload["samples"][0]["prediction"] = {}
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Unknown|unknown"):
        GroundTruthDataset.load(path)


def test_loader_rejects_empty_dataset(tmp_path):
    payload = valid_payload()
    payload["samples"] = []
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="empty"):
        GroundTruthDataset.load(path)
