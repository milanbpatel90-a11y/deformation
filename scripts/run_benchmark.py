#!/usr/bin/env python3
"""Standalone production measurement benchmark CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.evaluation.benchmark import default_report_name, run_benchmark
from backend.evaluation.dataset import GroundTruthDataset, load_csv
from backend.evaluation.report import print_report, write_report
from backend.pipeline import DeformationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate production measurements against hand-measured ground truth"
    )
    parser.add_argument("--dataset", required=True, help="Path to samples.json or strict CSV")
    parser.add_argument("--baseline", help="Optional baseline.json used for regression gating")
    parser.add_argument("--report", help="Output report path")
    parser.add_argument("--output-dir", default="reports/benchmark_outputs")
    parser.add_argument("--slack", type=float, default=0.25)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run the real pipeline twice per image and require identical measurements",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = load_csv(dataset_path) if dataset_path.suffix.lower() == ".csv" else GroundTruthDataset.load(dataset_path)

    def factory() -> DeformationPipeline:
        return DeformationPipeline()

    report, rows = run_benchmark(dataset, factory, dataset_path, args.output_dir)
    print_report(report)

    sha = os.getenv("GITHUB_SHA", "local")
    report_path = Path(args.report) if args.report else Path("reports") / default_report_name(sha)
    write_report(
        report,
        report_path,
        metadata={"dataset": str(dataset_path), "sha": sha, "samples": len(rows)},
    )
    print(f"Report: {report_path}")

    if args.check_determinism:
        from backend.evaluation.benchmark import resolve_image
        for sample in dataset.samples:
            image = resolve_image(dataset_path, sample.image)
            first = factory().run_from_images(
                image, output_path=Path(args.output_dir) / f"{sample.id}_det_a.glb"
            )["measurements"]
            second = factory().run_from_images(
                image, output_path=Path(args.output_dir) / f"{sample.id}_det_b.glb"
            )["measurements"]
            for field in ("frame_width", "bridge_width", "lens_width", "lens_height", "temple_length"):
                if first[field] != second[field]:
                    raise SystemExit(
                        f"Nondeterministic measurement for {sample.id}.{field}: "
                        f"{first[field]} != {second[field]}"
                    )

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_metrics = baseline.get("metrics", {}).get("overall", {})
        if not baseline_metrics:
            print("Baseline gate: not established; regression gate skipped.")
        else:
            for field, metrics in report.overall.items():
                base = baseline_metrics.get(field, {}).get("mae")
                if base is None:
                    raise SystemExit(f"baseline.json missing metrics.overall.{field}.mae")
                if metrics.mae > float(base) + args.slack:
                    raise SystemExit(
                        f"MAE regression: {field}={metrics.mae:.3f} > "
                        f"baseline {float(base):.3f} + slack {args.slack:.3f}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
