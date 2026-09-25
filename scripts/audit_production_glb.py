"""CLI audit for a production deformation GLB.

Example:
    python scripts/audit_production_glb.py output/model.glb \
        --frame-width 135 --lens-width 50 --lens-height 46 \
        --bridge-width 16 --temple-length 140
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.exporter.glb_validator import validate_glb
from backend.models import Measurements


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an exported production eyewear GLB")
    parser.add_argument("glb", type=Path)
    parser.add_argument("--frame-width", type=float, required=True)
    parser.add_argument("--lens-width", type=float, required=True)
    parser.add_argument("--lens-height", type=float, required=True)
    parser.add_argument("--bridge-width", type=float, required=True)
    parser.add_argument("--temple-length", type=float, required=True)
    parser.add_argument("--rim-thickness", type=float, default=1.2)
    parser.add_argument("--frame-tolerance-mm", type=float, default=5.0)
    args = parser.parse_args()

    measurements = Measurements(
        frame_width=args.frame_width,
        lens_width=args.lens_width,
        lens_height=args.lens_height,
        bridge_width=args.bridge_width,
        temple_length=args.temple_length,
        rim_thickness=args.rim_thickness,
    )

    report = validate_glb(
        args.glb,
        measurements,
        frame_tolerance_mm=args.frame_tolerance_mm,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
