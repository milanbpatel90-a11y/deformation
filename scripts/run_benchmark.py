#!/usr/bin/env python3
"""CLI for the production measurement benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from backend.evaluation.benchmark import default_report_name, run_benchmark
from backend.evaluation.dataset import GroundTruthDataset
from backend.evaluation.report import print_report, write_report
from backend.pipeline import DeformationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate production measurements against hand-measured ground truth")
    parser.add_argument("--dataset", required=True, help="Path to canonical samples.json")
    parser.add_argument("--baseline", help="Optional baseline.json used as a regression gate")
    parser.add_argument("--report", help="Output report path; defaults to reports/benchmark_<sha>.json")
    parser.add_argument("--output-dir", default="reports/benchmark_outputs", help="Directory for generated GLBs")
    parser.add_argument("--slack", type=float, default=0.25, help="Allowed MAE regression over baseline in mm")
    parser.add_argument("--check-determinism", action="store_true", help="Run each image twice and require identical measurements")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = GroundTruthDataset.load(dataset_path)

    def factory():
        return DeformationPipeline()

    report, rows = run_benchmark(dataset, factory, dataset_path, args.output_dir)
    print_report(report)

    sha = os.getenv("GITHUB_SHA", "local")
    report_path = Path(args.report or "reports") / default_report_name(sha)
    write_report(report, report_path, metadata={"dataset": str(dataset_path), "sha": sha, "samples": len(rows)})
    print(f"\nReport: {report_path}")

    if args.check_determinism:
        for sample in dataset.samples:
            image = __import__("backend.evaluation.benchmark", fromlist=["resolve_image"]).resolve_image(dataset_path, sample.image)
            first = factory().run_from_images(image, output_path=Path(args.output_dir) / f"{sample.id}_det_a.glb")["measurements"]
            second = factory().run_from_images(image, output_path=Path(args.output_dir) / f"{sample.id}_det_b.glb")["measurements"]
            for field in ("frame_width", "bridge_width", "lens_width", "lens_height", "temple_length"):
                if first[field] != second[field]:
                    raise SystemExit(f"Nondeterministic measurement for {sample.id}.{field}: {first[field]} != {second[field]}")

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_metrics = baseline.get("metrics", {}).get("overall", {})
        if not baseline_metrics:
            raise SystemExit("baseline.json has no metrics.overall; create it from a real benchmark run before gating CI")
        for field, metrics in report.overall.items():
            base = baseline_metrics.get(field, {}).get("mae")
            if base is None:
                raise SystemExit(f"baseline.json missing metrics.overall.{field}.mae")
            if metrics.mae > float(base) + args.slack:
                raise SystemExit(f"MAE regression: {field}={metrics.mae:.3f} > baseline {float(base):.3f} + slack {args.slack:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
