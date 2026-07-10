"""CLI entry point for deformation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


def main():
    parser = argparse.ArgumentParser(description="Defirmation — eyewear template deformation")
    sub = parser.add_subparsers(dest="command")

    # deform from images
    img_parser = sub.add_parser("deform", help="Deform from product images")
    img_parser.add_argument("--front", required=True, help="Front image path")
    img_parser.add_argument("--side", help="Side image path")
    img_parser.add_argument("--output", "-o", default="output/deformed.glb")
    img_parser.add_argument("--color", default="#d9a7a2")
    img_parser.add_argument("--template", help="Force template name")

    # deform from measurements JSON
    meas_parser = sub.add_parser("measurements", help="Deform from measurements JSON")
    meas_parser.add_argument("--input", "-i", required=True, help="Measurements JSON file")
    meas_parser.add_argument("--output", "-o", default="output/deformed.glb")
    meas_parser.add_argument("--template", default="geometric_metal")

    # generate template
    sub.add_parser("generate-template", help="Generate geometric_metal.glb")

    args = parser.parse_args()

    if args.command == "generate-template":
        from scripts.generate_template import main as gen
        gen()
        return

    pipeline = DeformationPipeline()

    if args.command == "deform":
        result = pipeline.run_from_images(
            args.front, args.side, args.output, args.color, args.template
        )
        print(json.dumps(result, indent=2))

    elif args.command == "measurements":
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        measurements = Measurements(**data)
        result = pipeline.run_from_measurements(measurements, args.output, args.template)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
