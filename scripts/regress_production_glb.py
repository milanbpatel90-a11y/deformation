"""Run deformation regression against the real production GLB.

Writes results even when deformation fails so baseline/fixed behavior can be
compared honestly.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import trimesh

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


def scene_metrics(path: Path) -> dict:
    scene = trimesh.load(path, force="scene", process=False)
    bounds = scene.bounds
    geoms = {}
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        geoms[name] = {
            "vertices": int(len(geom.vertices)),
            "triangles": int(len(geom.faces)),
            "bounds": geom.bounds.tolist(),
            "extents": geom.extents.tolist(),
            "finite": bool(np.isfinite(geom.vertices).all()),
            "watertight": bool(geom.is_watertight),
            "winding_consistent": bool(geom.is_winding_consistent),
        }
    return {
        "file_size_bytes": path.stat().st_size,
        "scene_bounds": bounds.tolist() if bounds is not None else None,
        "scene_extents": (bounds[1] - bounds[0]).tolist() if bounds is not None else None,
        "geometry_count": len(geoms),
        "geometries": geoms,
    }


def run_case(name: str, measurements: Measurements, output_dir: Path) -> dict:
    pipeline = DeformationPipeline(Path("templates"))
    out = output_dir / f"{name}.glb"
    result = {
        "name": name,
        "measurements": measurements.model_dump(mode="json"),
        "success": False,
    }
    try:
        payload = pipeline.run_from_measurements(
            measurements,
            out,
            template_name="geometric_metal",
        )
        result["success"] = True
        result["pipeline_result"] = payload
        result["output"] = scene_metrics(out)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        if out.exists():
            result["partial_output"] = scene_metrics(out)
    return result


def main() -> int:
    output_dir = Path("output/production_regression")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "identity_135mm",
            Measurements(
                frame_width=135.0,
                lens_width=50.0,
                lens_height=46.0,
                bridge_width=16.0,
                temple_length=135.0,
                rim_thickness=1.0,
                material=FrameMaterial.METAL,
                shape=FrameShape.GEOMETRIC,
                nose_pads=True,
            ),
        ),
        (
            "target_145mm",
            Measurements(
                frame_width=145.0,
                lens_width=52.0,
                lens_height=48.0,
                bridge_width=18.0,
                temple_length=140.0,
                rim_thickness=1.2,
                material=FrameMaterial.METAL,
                shape=FrameShape.GEOMETRIC,
                nose_pads=True,
            ),
        ),
    ]

    source = Path("templates/geometric_metal.glb")
    report = {
        "source": scene_metrics(source),
        "cases": [run_case(name, measurements, output_dir) for name, measurements in cases],
    }
    path = output_dir / "regression.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("SOURCE", json.dumps(report["source"], sort_keys=True))
    for case in report["cases"]:
        print("CASE", case["name"], "success", case["success"])
        if case["success"]:
            print(" output_extents", case["output"]["scene_extents"])
            print(" output_size", case["output"]["file_size_bytes"])
            print(" quality", json.dumps(case["pipeline_result"].get("quality"), sort_keys=True))
        else:
            print(" error", case.get("error"))
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
