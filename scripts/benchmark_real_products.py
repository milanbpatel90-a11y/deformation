"""Evaluate photo-to-template deformation against labeled real eyewear.

The runner intentionally refuses empty manifests and never treats generated or
synthetic images as product validation. See dataset/real_products/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from backend.pipeline import DeformationPipeline

FIELDS = ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length", "rim_thickness")


def summarize(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0, "mae_mm": None, "rmse_mm": None, "max_abs_error_mm": None}
    return {
        "count": len(errors),
        "mae_mm": sum(abs(value) for value in errors) / len(errors),
        "rmse_mm": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "max_abs_error_mm": max(abs(value) for value in errors),
    }


def run_manifest(manifest_path: Path, output_dir: Path, min_products: int = 100) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    products = payload.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Manifest must contain labeled real products; an empty fixture is not validation")
    ids = [item.get("id") for item in products if isinstance(item, dict)]
    if len(ids) != len(products) or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every product needs a unique, non-empty string id")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DeformationPipeline()
    absolute_errors = {field: [] for field in FIELDS}
    rows, failures = [], []
    for product in products:
        try:
            refs = product["measurements_mm"]
            if any(field not in refs for field in FIELDS):
                raise ValueError(f"measurements_mm must include all fields: {', '.join(FIELDS)}")
            raw_images = product["images"]
            if isinstance(raw_images, dict):
                raw_images = list(raw_images.values())
            if not isinstance(raw_images, list) or not 1 <= len(raw_images) <= 6:
                raise ValueError("images must contain 1–6 relative or absolute image paths")
            images = [(manifest_path.parent / item).resolve() if not Path(item).is_absolute() else Path(item)
                      for item in raw_images]
            missing = [str(item) for item in images if not item.is_file()]
            if missing:
                raise FileNotFoundError("Missing image files: " + ", ".join(missing))
            result = pipeline.run_from_multiple_images(
                images,
                output_dir / f"{product['id']}.glb",
                automatic_appearance=True,
            )
            measured = result["measurements"]
            errors = {field: float(measured[field]) - float(refs[field]) for field in FIELDS}
            for field, error in errors.items():
                absolute_errors[field].append(error)
            rows.append({
                "id": product["id"],
                "expected_template": product.get("expected_template"),
                "selected_template": result["template"],
                "template_match": product.get("expected_template") in (None, result["template"]),
                "measurement_errors_mm": errors,
                "acceptance": result["acceptance"],
                "output_glb": result["output_glb"],
                "manifest_json": result["manifest_json"],
            })
        except Exception as exc:  # Report all failures so the data owner can fix a batch.
            failures.append({"id": product["id"], "error": str(exc)})

    metrics = {field: summarize(values) for field, values in absolute_errors.items()}
    thresholds = payload.get("acceptance_criteria")
    mismatch = sum(not row["template_match"] for row in rows)
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "products_requested": len(products),
        "products_evaluated": len(rows),
        "failures": failures,
        "template_mismatches": mismatch,
        "metrics": metrics,
        "acceptance_criteria": thresholds,
        "products": rows,
    }
    criteria_passed = False
    if isinstance(thresholds, dict) and rows and not failures:
        field_limit = float(thresholds.get("max_mae_mm", -1))
        maximum_limit = float(thresholds.get("max_error_mm", -1))
        criteria_passed = (field_limit >= 0 and maximum_limit >= 0
                           and all(metric["mae_mm"] <= field_limit
                                   and metric["max_abs_error_mm"] <= maximum_limit
                                   for metric in metrics.values())
                           and mismatch == 0
                           and all(row["acceptance"]["status"] == "PASS" for row in rows))
    report["production_validation"] = {
        "passed": bool(len(products) >= min_products and criteria_passed),
        "minimum_products_required": min_products,
        "labeled_product_criteria_passed": criteria_passed,
        "note": "A separate exact self-intersection gate is required for full production acceptance.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/debug/real_products"))
    parser.add_argument("--min-products", type=int, default=100)
    parser.add_argument("--report", type=Path, default=Path("output/debug/real_products/report.json"))
    args = parser.parse_args()
    report = run_manifest(args.manifest, args.output_dir, args.min_products)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("products_requested", "products_evaluated", "template_mismatches", "metrics", "production_validation")},
                     indent=2))
    raise SystemExit(0 if report["production_validation"]["passed"] else 2)


if __name__ == "__main__":
    main()
