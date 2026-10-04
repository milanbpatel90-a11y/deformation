"""Strict loader for the canonical eyewear measurement ground-truth dataset."""

from __future__ import annotations

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
ALLOWED_SAMPLE_FIELDS = REQUIRED_SAMPLE_FIELDS


class DatasetValidationError(ValueError):
    """Raised when a benchmark dataset violates the canonical schema."""


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
        if not isinstance(payload["samples"], list):
            raise DatasetValidationError("samples must be an array")
        if not payload["samples"]:
            raise DatasetValidationError("Ground-truth dataset is empty")

        samples: list[GroundTruthSample] = []
        seen: set[str] = set()
        for index, raw in enumerate(payload["samples"]):
            if not isinstance(raw, dict):
                raise DatasetValidationError(f"samples[{index}] must be an object")
            unknown = set(raw) - ALLOWED_SAMPLE_FIELDS
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
            tol = _strict_measurement_map(raw["tolerance"], f"samples[{index}].tolerance")
            if any(value < 0 for value in tol.values()):
                raise DatasetValidationError(f"samples[{index}].tolerance cannot contain negative values")
            samples.append(GroundTruthSample(
                id=sample_id,
                image=str(raw["image"]),
                archetype=str(raw["archetype"]),
                source=str(raw["source"]),
                measured_by=str(raw["measured_by"]),
                ground_truth=gt,
                tolerance=tol,
            ))
        return cls(version=1, units="mm", samples=tuple(samples))


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
