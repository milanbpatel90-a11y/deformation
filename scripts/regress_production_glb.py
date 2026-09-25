"""Run deformation regression against the real production GLB.

Writes results even when deformation fails so baseline/fixed behavior can be
compared honestly. Physical acceptance is based on logical components, not on
whole-scene bounds.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


_REQUIRED = (
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftLens",
    "RightLens",
    "LeftTemple",
    "RightTemple",
)


def _union_bounds(scene: trimesh.Scene, names: list[str]) -> np.ndarray | None:
    selected = [
        np.asarray(scene.geometry[name].bounds, dtype=np.float64)
        for name in names
        if name in scene.geometry
    ]
    if not selected:
        return None
    lows = np.vstack([bounds[0] for bounds in selected])
    highs = np.vstack([bounds[1] for bounds in selected])
    return np.array([lows.min(axis=0), highs.max(axis=0)], dtype=np.float64)


def scene_metrics(path: Path) -> dict:
    scene = trimesh.load(path, force="scene", process=False)
    bounds = scene.bounds
    geoms = {}

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        faces = np.asarray(geom.faces)
        verts = np.asarray(geom.vertices)
        degenerate = 0
        if len(faces):
            tris = verts[faces]
            area2 = np.linalg.norm(
                np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                axis=1,
            )
            degenerate = int(np.count_nonzero(area2 <= 1e-12))

        normals = np.asarray(geom.vertex_normals) if len(verts) else np.empty((0, 3))
        normal_lengths = np.linalg.norm(normals, axis=1) if len(normals) else np.empty(0)
        material = getattr(geom.visual, "material", None)

        geoms[name] = {
            "vertices": int(len(verts)),
            "triangles": int(len(faces)),
            "bounds": geom.bounds.tolist(),
            "extents": geom.extents.tolist(),
            "finite": bool(np.isfinite(verts).all()),
            "watertight": bool(geom.is_watertight),
            "winding_consistent": bool(geom.is_winding_consistent),
            "degenerate_triangles": degenerate,
            "normals_finite": bool(np.isfinite(normals).all()) if len(normals) else True,
            "zero_normals": int(np.count_nonzero(normal_lengths <= 1e-12)) if len(normals) else 0,
            "material_name": getattr(material, "name", None),
        }

    frame_bounds = _union_bounds(scene, ["Bridge", "LeftRim", "RightRim"])
    if frame_bounds is None and "Frame" in scene.geometry:
        frame_bounds = np.asarray(scene.geometry["Frame"].bounds, dtype=np.float64)

    measured = {}
    if frame_bounds is not None:
        measured["frame_width_mm"] = float((frame_bounds[1, 0] - frame_bounds[0, 0]) * 1000.0)
        measured["frame_height_mm"] = float((frame_bounds[1, 1] - frame_bounds[0, 1]) * 1000.0)

    if "Bridge" in scene.geometry:
        measured["bridge_width_mm"] = float(scene.geometry["Bridge"].extents[0] * 1000.0)

    for side in ("Left", "Right"):
        lens_name = f"{side}Lens"
        temple_name = f"{side}Temple"
        key = side.lower()
        if lens_name in scene.geometry:
            measured[f"{key}_lens_width_mm"] = float(scene.geometry[lens_name].extents[0] * 1000.0)
            measured[f"{key}_lens_height_mm"] = float(scene.geometry[lens_name].extents[1] * 1000.0)
        if temple_name in scene.geometry:
            measured[f"{key}_temple_length_mm"] = float(
                np.max(scene.geometry[temple_name].extents) * 1000.0
            )

    return {
        "file_size_bytes": path.stat().st_size,
        "scene_bounds": bounds.tolist() if bounds is not None else None,
        "scene_extents": (bounds[1] - bounds[0]).tolist() if bounds is not None else None,
        "geometry_count": len(geoms),
        "geometries": geoms,
        "measured_dimensions": measured,
        "required_components": {name: name in scene.geometry for name in _REQUIRED},
        "independent_lenses": (
            "LeftLens" in scene.geometry
            and "RightLens" in scene.geometry
            and scene.geometry["LeftLens"] is not scene.geometry["RightLens"]
        ),
        "independent_temples": (
            "LeftTemple" in scene.geometry
            and "RightTemple" in scene.geometry
            and scene.geometry["LeftTemple"] is not scene.geometry["RightTemple"]
        ),
    }


def _dimension_errors(metrics: dict, measurements: Measurements) -> dict:
    measured = metrics["measured_dimensions"]
    targets = {
        "frame_width": ("frame_width_mm", measurements.frame_width),
        "bridge_width": ("bridge_width_mm", measurements.bridge_width),
        "left_lens_width": ("left_lens_width_mm", measurements.lens_width),
        "right_lens_width": ("right_lens_width_mm", measurements.lens_width),
        "left_lens_height": ("left_lens_height_mm", measurements.lens_height),
        "right_lens_height": ("right_lens_height_mm", measurements.lens_height),
        "left_temple_length": ("left_temple_length_mm", measurements.temple_length),
        "right_temple_length": ("right_temple_length_mm", measurements.temple_length),
    }
    errors = {}
    for key, (measured_key, target) in targets.items():
        value = measured.get(measured_key)
        errors[key] = None if value is None else float(value - target)
    return errors


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
        result["dimension_errors_mm"] = _dimension_errors(result["output"], measurements)
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
            print(" measured_dimensions", json.dumps(case["output"]["measured_dimensions"], sort_keys=True))
            print(" dimension_errors_mm", json.dumps(case["dimension_errors_mm"], sort_keys=True))
            print(" required_components", json.dumps(case["output"]["required_components"], sort_keys=True))
            print(" independent_lenses", case["output"]["independent_lenses"])
            print(" independent_temples", case["output"]["independent_temples"])
            print(" quality", json.dumps(case["pipeline_result"].get("quality"), sort_keys=True))
        else:
            print(" error", case.get("error"))
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
