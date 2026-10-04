"""Measurement benchmark metrics with explicit error-sign conventions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping

from backend.evaluation.dataset import FIELDS


@dataclass(frozen=True)
class FieldMetrics:
    mae: float
    median_ae: float
    p90_ae: float
    max_ae: float
    percent_mae: float
    bias: float
    pass_rate: float
    count: int


@dataclass(frozen=True)
class EvaluationReport:
    overall: dict[str, FieldMetrics]
    by_archetype: dict[str, dict[str, FieldMetrics]]
    worst_cases: dict[str, list[dict[str, object]]]


def evaluate_predictions(
    rows: Iterable[Mapping[str, object]],
    fields: tuple[str, ...] = FIELDS,
    worst_n: int = 5,
) -> EvaluationReport:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    for row_index, row in enumerate(rows):
        for field in fields:
            if field not in row:
                raise ValueError(f"Prediction row {row_index} missing field '{field}'")
            if not isinstance(row[field], (int, float)) or isinstance(row[field], bool):
                raise ValueError(f"Prediction row {row_index} field '{field}' must be numeric")
            gt_key = f"gt_{field}"
            if gt_key not in row:
                raise ValueError(f"Prediction row {row_index} missing field '{gt_key}'")
            if not isinstance(row[gt_key], (int, float)) or isinstance(row[gt_key], bool):
                raise ValueError(f"Prediction row {row_index} field '{gt_key}' must be numeric")
            tol_key = f"tolerance_{field}"
            if tol_key not in row:
                raise ValueError(f"Prediction row {row_index} missing field '{tol_key}'")

    overall = {field: _field_metrics(rows, field) for field in fields}
    archetypes = sorted({str(row.get("archetype", "unknown")) for row in rows})
    by_archetype = {
        archetype: {
            field: _field_metrics([r for r in rows if str(r.get("archetype", "unknown")) == archetype], field)
            for field in fields
        }
        for archetype in archetypes
    }
    worst_cases = {
        field: sorted(
            [
                {
                    "id": str(row.get("id", "unknown")),
                    "gt": float(row[f"gt_{field}"]),
                    "pred": float(row[field]),
                    "err": float(row[field]) - float(row[f"gt_{field}"]),
                    "archetype": str(row.get("archetype", "unknown")),
                }
                for row in rows
            ],
            key=lambda item: abs(float(item["err"])),
            reverse=True,
        )[:worst_n]
        for field in fields
    }
    return EvaluationReport(overall=overall, by_archetype=by_archetype, worst_cases=worst_cases)


def _field_metrics(rows: list[Mapping[str, object]], field: str) -> FieldMetrics:
    signed = [float(row[field]) - float(row[f"gt_{field}"]) for row in rows]
    absolute = [abs(error) for error in signed]
    gt = [abs(float(row[f"gt_{field}"])) for row in rows]
    tolerances = [float(row[f"tolerance_{field}"]) for row in rows]
    count = len(rows)
    mae = sum(absolute) / count
    median_ae = median(absolute)
    ordered = sorted(absolute)
    rank = max(0, min(count - 1, int(__import__("math").ceil(0.90 * count) - 1))
    p90_ae = ordered[rank]
    max_ae = max(absolute)
    mean_gt = sum(gt) / count
    percent_mae = (mae / mean_gt * 100.0) if mean_gt else 0.0
    pass_rate = sum(error <= tolerance for error, tolerance in zip(absolute, tolerances)) / count
    return FieldMetrics(
        mae=mae,
        median_ae=median_ae,
        p90_ae=p90_ae,
        max_ae=max_ae,
        percent_mae=percent_mae,
        bias=sum(signed) / count,
        pass_rate=pass_rate,
        count=count,
    )
