"""Strict JSON/CSV loader for canonical eyewear measurement ground truth."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIELDS = (
    "frame_width",
    "bridge_width",
    "lens_width",
    "lens_height",
    "temple_length",
)
REQUIRED_SAMPLE_FIELDS = {"id", "image", "archetype", "source", "measured_by", "ground_truth", "tolerance"}
ALLOWED_TOP_FIELDS = {"version", "units", "samples"}


class DatasetValidationError(ValueError):
    """Raised when benchmark data violates the canonical schema."""


@dataclass(frozen=True)
class GroundTruthSample:
    id: str
    image: str
    archetype: str
    source: str
    measured_by: str
    ground_truth: dict[str, float]
    tolerance: dict[str, float]


@dataclass(frozen=True)
class GroundTruthDataset:
    version: int
    units: str
    samples: tuple[GroundTruthSample, ...]

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruthDataset":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetValidationError(f"Dataset file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise DatasetValidationError("Dataset root must be an object")
        unknown = set(payload) - ALLOWED_TOP_FIELDS
        missing = ALLOWED_TOP_FIELDS - set(payload)
        if unknown:
            raise DatasetValidationError(f"Unknown dataset fields: {sorted(unknown)}")
        if missing:
            raise DatasetValidationError(f"Missing dataset fields: {sorted(missing)}")
        if payload["version"] != 1:
            raise DatasetValidationError("Only dataset version 1 is supported")
        if payload["units"] != "mm":
            raise DatasetValidationError("Ground-truth units must be exactly 'mm'")
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise DatasetValidationError("Ground-truth dataset is empty")

        samples: list[GroundTruthSample] = []
        seen: set[str] = set()
        for index, raw in enumerate(payload["samples"]):
            if not isinstance(raw, dict):
                raise DatasetValidationError(f"samples[{index}] must be an object")
            unknown = set(raw) - REQUIRED_SAMPLE_FIELDS
            missing = REQUIRED_SAMPLE_FIELDS - set(raw)
            if unknown:
                raise DatasetValidationError(f"samples[{index}] unknown fields: {sorted(unknown)}")
            if missing:
                raise DatasetValidationError(f"samples[{index}] missing fields: {sorted(missing)}")
            sample_id = raw["id"]
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise DatasetValidationError(f"samples[{index}].id must be a non-empty string")
            if sample_id in seen:
                raise DatasetValidationError(f"Duplicate sample id: {sample_id}")
            seen.add(sample_id)
            gt = _strict_measurement_map(raw["ground_truth"], f"samples[{index}].ground_truth")
            tolerance = _strict_measurement_map(raw["tolerance"], f"samples[{index}].tolerance")
            if any(value < 0 for value in tolerance.values()):
                raise DatasetValidationError(f"samples[{index}].tolerance cannot contain negative values")
            samples.append(
                GroundTruthSample(
                    id=sample_id,
                    image=str(raw["image"]),
                    archetype=str(raw["archetype"]),
                    source=str(raw["source"]),
                    measured_by=str(raw["measured_by"]),
                    ground_truth=gt,
                    tolerance=tolerance,
                )
            )
        return cls(version=1, units="mm", samples=tuple(samples))

    @classmethod
    def from_csv(cls, path: str | Path) -> "GroundTruthDataset":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise DatasetValidationError("CSV has no header")
                required = {
                    "id", "image", "archetype", "source", "measured_by", "units",
                    *FIELDS,
                    *(f"{field}_tolerance" for field in FIELDS),
                }
                unknown = set(reader.fieldnames) - required
                missing = required - set(reader.fieldnames)
                if unknown:
                    raise DatasetValidationError(f"Unknown CSV fields: {sorted(unknown)}")
                if missing:
                    raise DatasetValidationError(f"Missing CSV fields: {sorted(missing)}")

                payload_samples: list[dict[str, Any]] = []
                for line, row in enumerate(reader, start=2):
                    if row.get("units") != "mm":
                        raise DatasetValidationError(f"CSV row {line} units must be exactly 'mm'")
                    gt = {field: _csv_float(row.get(field), f"CSV row {line}.{field}") for field in FIELDS}
                    tolerance = {
                        field: _csv_float(row.get(f"{field}_tolerance"), f"CSV row {line}.{field}_tolerance")
                        for field in FIELDS
                    }
                    if any(value < 0 for value in tolerance.values()):
                        raise DatasetValidationError(f"CSV row {line} tolerance cannot be negative")
                    payload_samples.append({
                        "id": row["id"],
                        "image": row["image"],
                        "archetype": row["archetype"],
                        "source": row["source"],
                        "measured_by": row["measured_by"],
                        "ground_truth": gt,
                        "tolerance": tolerance,
                    })
        except FileNotFoundError as exc:
            raise DatasetValidationError(f"CSV file not found: {path}") from exc

        if not payload_samples:
            raise DatasetValidationError("Ground-truth CSV is empty")
        temp_payload = {"version": 1, "units": "mm", "samples": payload_samples}
        temp_path = path.with_suffix(".validated.json")
        try:
            temp_path.write_text(json.dumps(temp_payload), encoding="utf-8")
            return cls.load(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


def load_csv(path: str | Path) -> GroundTruthDataset:
    """Backward-compatible function form of the strict CSV importer."""
    return GroundTruthDataset.from_csv(path)


def _csv_float(value: str | None, label: str) -> float:
    if value is None or value == "":
        raise DatasetValidationError(f"{label} is required")
    try:
        return float(value)
    except ValueError as exc:
        raise DatasetValidationError(f"{label} must be numeric") from exc


def _strict_measurement_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{label} must be an object")
    unknown = set(value) - set(FIELDS)
    missing = set(FIELDS) - set(value)
    if unknown:
        raise DatasetValidationError(f"{label} unknown fields: {sorted(unknown)}")
    if missing:
        raise DatasetValidationError(f"{label} missing fields: {sorted(missing)}")
    result: dict[str, float] = {}
    for field in FIELDS:
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DatasetValidationError(f"{label}.{field} must be numeric")
        result[field] = float(raw)
    return result
