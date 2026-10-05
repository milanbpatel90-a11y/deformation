"""JSON and console reporting for measurement benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from backend.evaluation.metrics import EvaluationReport


def report_to_dict(report: EvaluationReport) -> dict:
    return asdict(report)


def write_report(report: EvaluationReport, path: str | Path, *, metadata: dict | None = None) -> None:
    payload = {"version": 1, "metrics": report_to_dict(report)}
    if metadata:
        payload["metadata"] = metadata
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_report(report: EvaluationReport) -> None:
    print("Measurement benchmark")
    print("field               MAE    median    p90    max   %MAE   bias   pass")
    for field, metrics in report.overall.items():
        print(f"{field:16s} {metrics.mae:6.2f} {metrics.median_ae:8.2f} {metrics.p90_ae:6.2f} {metrics.max_ae:6.2f} {metrics.percent_mae:6.2f}% {metrics.bias:+6.2f} {metrics.pass_rate:6.1%}")
    print("\nWorst cases")
    for field, cases in report.worst_cases.items():
        print(f"{field}:")
        for case in cases:
            print(f"  {case['id']}: gt={case['gt']:.2f} pred={case['pred']:.2f} err={case['err']:+.2f} archetype={case['archetype']}")
    print("\nBy archetype")
    for archetype, fields in report.by_archetype.items():
        print(archetype)
        for field, metrics in fields.items():
            print(f"  {field:16s} MAE={metrics.mae:.2f} bias={metrics.bias:+.2f} pass={metrics.pass_rate:.1%}")
