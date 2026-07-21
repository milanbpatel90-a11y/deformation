"""CLI entry point for the glasses3d pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from pipeline.classify import classify_frame
from pipeline.deform import deform_template
from pipeline.export import export_glb
from pipeline.landmarks import detect_landmarks
from pipeline.materials import build_materials
from pipeline.measurements import extract_measurements
from pipeline.preprocess import preprocess
from pipeline.template import load_template
from pipeline.texture import create_texture_atlas
from pipeline.validate import validate_model


def run(front: str, side: str | None, output: str, frame_width_mm: float | None = None, debug: bool = False) -> dict:
    if not side: warnings.warn("Side view missing; using default temple length", UserWarning)
    debug_dir = Path("debug") if debug else None
    if debug_dir: debug_dir.mkdir(exist_ok=True)
    logging.info("1/11 preprocessing images")
    data = preprocess(front, side)
    logging.info("2/11 classifying frame")
    classification = classify_frame(data["front_masks"])
    logging.info("3/11 detecting landmarks")
    landmarks = detect_landmarks(data["front_masks"], data["side_masks"])
    logging.info("4/11 extracting measurements")
    measurements = extract_measurements(landmarks, frame_width_mm)
    logging.info("5/11 retrieving %s template", classification["frame_type"])
    template = load_template(classification["frame_type"], Path(__file__).parent / "templates")
    logging.info("6/11 deforming template")
    model = deform_template(template, measurements)
    logging.info("7/11 generating materials")
    materials = build_materials(data["front_rgba"], data["front_masks"])
    texture_path = (debug_dir or Path(output).parent) / "texture_atlas.png"
    logging.info("8/11 generating texture atlas")
    create_texture_atlas(data["front_rgba"], texture_path)
    logging.info("9/11 validating model")
    report = validate_model(model["parts"], data["front_masks"]["frame_front"], str(texture_path))
    if not report["passed"]: raise RuntimeError(f"Validation failed: {json.dumps(report)}")
    logging.info("10/11 exporting GLB")
    export_glb(model, output, materials)
    result = {"classification": classification, "landmarks": landmarks, "measurements": measurements.to_dict(), "validation": report, "output": str(Path(output).resolve())}
    if debug: (debug_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("11/11 complete: %s", output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True); parser.add_argument("--side"); parser.add_argument("--frame-width-mm", type=float); parser.add_argument("--out", required=True); parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.front, args.side, args.out, args.frame_width_mm, args.debug)


if __name__ == "__main__": main()
