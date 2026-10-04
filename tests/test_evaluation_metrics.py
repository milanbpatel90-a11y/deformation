import pytest

from backend.evaluation.metrics import evaluate_predictions

FIELDS = ("frame_width", "bridge_width", "lens_width", "lens_height", "temple_length")


def row(pred, gt=None, tol=0.5, sample_id="s1", archetype="wayfarer"):
    gt = gt or {field: 100.0 for field in FIELDS}
    return {
        "id": sample_id,
        "archetype": archetype,
        **{field: float(pred.get(field, gt[field])) for field in FIELDS},
        **{f"gt_{field}": float(gt[field]) for field in FIELDS},
        **{f"tolerance_{field}": tol for field in FIELDS},
    }


def test_perfect_predictions_are_zero_error():
    report = evaluate_predictions([row({field: 100.0 for field in FIELDS}, sample_id=f"s{i}") for i in range(3)])
    metrics = report.overall["frame_width"]
    assert metrics.mae == 0.0
    assert metrics.bias == 0.0
    assert metrics.pass_rate == 1.0


def test_constant_positive_offset_exposes_bias_and_fails_tolerance():
    report = evaluate_predictions([row({field: 102.0 for field in FIELDS}, sample_id=f"s{i}") for i in range(3)])
    metrics = report.overall["frame_width"]
    assert metrics.mae == 2.0
    assert metrics.bias == 2.0
    assert metrics.max_ae == 2.0
    assert metrics.pass_rate == 0.0


def test_outlier_separates_median_from_mae():
    rows = [
        row({field: 100.0 for field in FIELDS}, sample_id="a"),
        row({field: 101.0 for field in FIELDS}, sample_id="b"),
        row({field: 100.0 for field in FIELDS}, sample_id="c"),
        row({field: 150.0 for field in FIELDS}, sample_id="outlier"),
    ]
    metrics = evaluate_predictions(rows).overall["frame_width"]
    assert metrics.max_ae == 50.0
    assert metrics.median_ae == 0.5
    assert metrics.mae == 12.75


def test_manual_mae_fixture_catches_metric_arithmetic():
    gt = {field: 100.0 for field in FIELDS}
    rows = []
    for i, value in enumerate([99.0, 102.0, 104.0]):
        pred = {field: 100.0 for field in FIELDS}
        pred["frame_width"] = value
        rows.append(row(pred, gt=gt, sample_id=f"m{i}"))
    metrics = evaluate_predictions(rows).overall["frame_width"]
    assert metrics.mae == pytest.approx(7.0 / 3.0)
    assert metrics.bias == pytest.approx(5.0 / 3.0)


def test_empty_dataset_raises():
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions([])


def test_missing_prediction_field_raises():
    bad = row({field: 100.0 for field in FIELDS})
    del bad["bridge_width"]
    with pytest.raises(ValueError, match="bridge_width"):
        evaluate_predictions([bad])


def test_missing_ground_truth_field_raises():
    bad = row({field: 100.0 for field in FIELDS})
    del bad["gt_lens_width"]
    with pytest.raises(ValueError, match="gt_lens_width"):
        evaluate_predictions([bad])
