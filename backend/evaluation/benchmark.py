"""Run the real production pipeline against ground-truth measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from backend.evaluation.dataset import GroundTruthDataset
from backend.evaluation.metrics import EvaluationReport, evaluate_predictions

FIELDS = ("frame_width", "bridge_width", "lens_width", "lens_height", "temple_length")


def resolve_image(dataset_path: Path, image: str) -> Path:
    path = Path(image)
    if not path.is_absolute():
        path = dataset_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth image not found: {path}")
    return path


def run_benchmark(
    dataset: GroundTruthDataset,
    pipeline_factory: Callable[[], object],
    dataset_path: str | Path,
    output_dir: str | Path,
) -> tuple[EvaluationReport, list[dict[str, object]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for sample in dataset.samples:
        image_path = resolve_image(Path(dataset_path), sample.image)
        pipeline = pipeline_factory()
        output_path = output_dir / f"{sample.id}.glb"
        result = pipeline.run_from_images(image_path, output_path=output_path)
        prediction = result.get("measurements")
        if not isinstance(prediction, dict):
            raise ValueError(f"Pipeline result for '{sample.id}' has no measurements dict")
        row: dict[str, object] = {
            "id": sample.id,
            "archetype": sample.archetype,
            "source": sample.source,
            "measured_by": sample.measured_by,
        }
        for field in FIELDS:
            if field not in prediction:
                raise ValueError(f"Pipeline measurement for '{sample.id}' missing field '{field}'")
            row[field] = float(prediction[field])
            row[f"gt_{field}"] = sample.ground_truth[field]
            row[f"tolerance_{field}"] = sample.tolerance[field]
        rows.append(row)

    return evaluate_predictions(rows, fields=FIELDS), rows


def default_report_name(commit: str) -> str:
    return f"benchmark_{commit[:12]}.json"


def short_sha(path: str | Path) -> str:
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:12]
