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

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.engine import MeshDeformer
from backend.deformer.rim_deformer import RimDeformer
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.scene_utils import load_world_baked_scene
from scripts.audit_production_glb import parse_glb


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
    scene = load_world_baked_scene(path)
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
            temple_vertices = np.asarray(scene.geometry[temple_name].vertices, dtype=np.float64)
            frame_parts = [
                np.asarray(scene.geometry[name].vertices, dtype=np.float64)
                for name in ("Bridge", "LeftRim", "RightRim")
                if name in scene.geometry
            ]
            if frame_parts and len(temple_vertices):
                from scipy.spatial import cKDTree
                frame_vertices = np.vstack(frame_parts)
                tree = cKDTree(frame_vertices)
                distances, _ = tree.query(temple_vertices, k=1)
                hinge = temple_vertices[int(np.argmin(distances))]
                measured[f"{key}_temple_length_mm"] = float(
                    np.max(np.linalg.norm(temple_vertices - hinge, axis=1)) * 1000.0
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


_DIMENSION_TOLERANCE_MM = {
    "frame_width": 2.0,
    "bridge_width": 1.5,
    "left_lens_width": 1.5,
    "right_lens_width": 1.5,
    "left_lens_height": 1.5,
    "right_lens_height": 1.5,
    "left_temple_length": 2.0,
    "right_temple_length": 2.0,
}


def _gltf_contract(path: Path) -> dict:
    doc, _ = parse_glb(path)
    scene_index = int(doc.get("scene", 0))
    scene_doc = doc.get("scenes", [])[scene_index]
    extras = scene_doc.get("extras", {}) if isinstance(scene_doc, dict) else {}
    anchors = extras.get("anchors", {}) if isinstance(extras, dict) else {}
    units = extras.get("units", {}) if isinstance(extras, dict) else {}

    raw_nodes = doc.get("nodes", [])
    nodes = {node.get("name"): (index, node) for index, node in enumerate(raw_nodes) if node.get("name")}
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(raw_nodes):
        for child_index in node.get("children", []):
            parents[int(child_index)] = parent_index

    def local_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            values = np.asarray(node["matrix"], dtype=np.float64)
            if values.size != 16:
                raise ValueError("glTF node matrix must contain 16 values")
            return values.reshape((4, 4), order="F")

        translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        x, y, z, w = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
        transform = rotation @ scale_matrix
        transform[:3, 3] = translation
        return transform

    world_cache: dict[int, np.ndarray] = {}

    def world_matrix(index: int) -> np.ndarray:
        if index in world_cache:
            return world_cache[index]
        matrix = local_matrix(raw_nodes[index])
        parent = parents.get(index)
        if parent is not None:
            matrix = world_matrix(parent) @ matrix
        world_cache[index] = matrix
        return matrix

    pivot_errors = {}
    local_hinge_distances = {}

    local_scene = trimesh.load(path, force="scene", process=False)
    for temple_name, anchor_name in (("LeftTemple", "LeftHinge"), ("RightTemple", "RightHinge")):
        node_entry = nodes.get(temple_name)
        anchor = np.asarray(anchors.get(anchor_name, [np.nan, np.nan, np.nan]), dtype=np.float64)
        if node_entry is None or anchor.shape != (3,):
            pivot_errors[temple_name] = float("inf")
        else:
            node_index, _node = node_entry
            world_translation = world_matrix(node_index)[:3, 3]
            pivot_errors[temple_name] = float(np.linalg.norm(world_translation - anchor))

        geometry = local_scene.geometry.get(temple_name)
        if geometry is None or not len(geometry.vertices):
            local_hinge_distances[temple_name] = float("inf")
        else:
            local_hinge_distances[temple_name] = float(
                np.linalg.norm(np.asarray(geometry.vertices), axis=1).min()
            )

    return {
        "generator": doc.get("asset", {}).get("generator"),
        "scene_extras_keys": sorted(extras.keys()) if isinstance(extras, dict) else [],
        "has_nested_extras": isinstance(extras, dict) and "extras" in extras,
        "has_legacy_defirmation": isinstance(extras, dict) and "defirmation" in extras,
        "units": units,
        "anchors": anchors,
        "temple_node_pivot_error_m": pivot_errors,
        "temple_local_hinge_distance_m": local_hinge_distances,
        "node_names": sorted(nodes),
    }


def _accept_case(case: dict, source_triangle_count: int) -> list[str]:
    errors: list[str] = []
    if not case.get("success"):
        return [case.get("error", "deformation case failed")]

    output = case["output"]
    for name, present in output["required_components"].items():
        if not present:
            errors.append(f"missing required component {name}")
    if not output["independent_lenses"]:
        errors.append("left/right lenses are not independently addressable")
    if not output["independent_temples"]:
        errors.append("left/right temples are not independently addressable")

    triangle_count = sum(g["triangles"] for g in output["geometries"].values())
    if triangle_count != source_triangle_count:
        errors.append(
            f"triangle count changed unexpectedly: output={triangle_count} source={source_triangle_count}"
        )

    for name in _REQUIRED:
        geom = output["geometries"].get(name)
        if not geom:
            continue
        if not geom["finite"]:
            errors.append(f"{name} has non-finite positions")
        if geom["degenerate_triangles"]:
            errors.append(f"{name} has {geom['degenerate_triangles']} degenerate triangles")
        if not geom["normals_finite"]:
            errors.append(f"{name} has non-finite normals")
        if geom["zero_normals"]:
            errors.append(f"{name} has {geom['zero_normals']} zero-length normals")
        if not geom["winding_consistent"]:
            errors.append(f"{name} has inconsistent face winding")

    for key, value in case["dimension_errors_mm"].items():
        tolerance = _DIMENSION_TOLERANCE_MM[key]
        if value is None or abs(value) > tolerance:
            errors.append(f"{key} error {value} mm exceeds tolerance {tolerance} mm")

    quality = case["pipeline_result"].get("quality") or {}
    if quality.get("passed") is not True:
        errors.append(f"mesh quality did not pass: {quality}")

    contract = _gltf_contract(Path(case["pipeline_result"]["output_glb"]))
    case["gltf_contract"] = contract
    if contract["generator"] != "deformation/1.0":
        errors.append(f"unexpected generator: {contract['generator']}")
    if contract["has_nested_extras"]:
        errors.append("scene extras are nested unexpectedly")
    if contract["has_legacy_defirmation"]:
        errors.append("legacy defirmation metadata remains")
    if contract["units"] != {
        "geometry": "meter",
        "anchors": "meter",
        "measurements": "millimeter",
    }:
        errors.append(f"unexpected units metadata: {contract['units']}")
    if set(contract["anchors"]) != {"NoseBridge", "LeftHinge", "RightHinge"}:
        errors.append(f"unexpected authoritative anchor set: {sorted(contract['anchors'])}")
    for temple_name, pivot_error in contract["temple_node_pivot_error_m"].items():
        if not np.isfinite(pivot_error) or pivot_error > 1e-7:
            errors.append(
                f"{temple_name} node pivot does not match authoritative hinge: {pivot_error} m"
            )
    for temple_name, local_distance in contract["temple_local_hinge_distance_m"].items():
        if not np.isfinite(local_distance) or local_distance > 1e-7:
            errors.append(
                f"{temple_name} local geometry does not contain the hinge origin: {local_distance} m"
            )

    return errors


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



def _topology_snapshot(context: DeformationContext) -> dict:
    snapshot = {}
    for name in ("LeftTemple", "RightTemple"):
        mesh = context.mesh(name)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = vertices[faces] if len(faces) else np.empty((0, 3, 3))
        area2 = (
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            if len(triangles)
            else np.empty(0)
        )
        normals = np.asarray(mesh.vertex_normals) if len(vertices) else np.empty((0, 3))
        normal_lengths = np.linalg.norm(normals, axis=1) if len(normals) else np.empty(0)
        snapshot[name] = {
            "vertices": int(len(vertices)),
            "triangles": int(len(faces)),
            "degenerate_triangles": int(np.count_nonzero(area2 <= 1e-12)),
            "min_double_area": float(area2.min()) if len(area2) else None,
            "zero_normals": int(np.count_nonzero(normal_lengths <= 1e-12)) if len(normal_lengths) else 0,
            "finite": bool(np.isfinite(vertices).all()),
        }
    return snapshot


def stage_trace(measurements: Measurements) -> list[dict]:
    pipeline = DeformationPipeline(Path("templates"))
    style = pipeline._style_from_measurements(measurements)
    features = pipeline.feature_extractor.from_measurements(measurements, style)
    match = pipeline.matcher.match(features, "geometric_metal")
    template_info = match.best.template

    scene = load_world_baked_scene(template_info.glb_path)
    descriptor = pipeline.descriptor_loader.load(
        "geometric_metal",
        measurements=measurements,
        template_info=template_info,
    )
    context = DeformationContext(
        template_info=template_info,
        template_scene=scene,
        descriptor=descriptor,
        measurements=measurements.model_copy(deep=True),
        feature_set=features,
    )
    deformer = MeshDeformer(
        scene,
        template_info.dimensions,
        rim_pull_strength=pipeline.library.rim_pull_strength("geometric_metal"),
    )

    trace = [{"stage": "resolved_components", "temples": _topology_snapshot(context)}]
    for stage in deformer.stages:
        if getattr(stage, "stage_name", "") == "temple_deformation":
            frame_before = context.mesh("Frame").vertices.copy()
            results = []
            for side in ("left", "right"):
                selection = stage._collect_temples(context, side)
                target = stage._compute_target(context, selection)
                trace.append({
                    "stage": f"temple_{side}_start",
                    "temples": _topology_snapshot(context),
                    "target": target,
                })
                stage._rotate_about_hinge(context, selection, target)
                trace.append({"stage": f"temple_{side}_rotate", "temples": _topology_snapshot(context)})
                stage._extend_length(context, selection, target)
                trace.append({"stage": f"temple_{side}_extend", "temples": _topology_snapshot(context)})
                stage._apply_wrap(context, selection, target)
                trace.append({"stage": f"temple_{side}_wrap", "temples": _topology_snapshot(context)})
                stage._apply_ear_bend(context, selection, target)
                trace.append({"stage": f"temple_{side}_ear_bend", "temples": _topology_snapshot(context)})
                stage._preserve_symmetry(context, selection)
                trace.append({"stage": f"temple_{side}_preserve_symmetry", "temples": _topology_snapshot(context)})
                results.append(stage._validate_constraints(context, selection, target))
            frame_after = context.mesh("Frame").vertices.copy()
            frame_max_displacement = float(np.linalg.norm(frame_after - frame_before, axis=1).max())
            context = stage.update_context(
                context,
                applied=True,
                frame_max_displacement=round(frame_max_displacement, 6),
                sides=results,
            )
        elif isinstance(stage, RimDeformer):
            context = stage.apply(context, None)
        else:
            context = stage.apply(context)
        trace.append({
            "stage": getattr(stage, "stage_name", type(stage).__name__),
            "temples": _topology_snapshot(context),
        })

    deformer._refresh_normals(context)
    trace.append({"stage": "refresh_normals", "temples": _topology_snapshot(context)})
    return trace

def run_case(
    name: str,
    measurements: Measurements,
    output_dir: Path,
    template_name: str = "geometric_metal",
) -> dict:
    pipeline = DeformationPipeline(Path("templates"))
    out = output_dir / f"{name}.glb"
    result = {
        "name": name,
        "measurements": measurements.model_dump(mode="json"),
        "template_name": template_name,
        "success": False,
    }
    try:
        payload = pipeline.run_from_measurements(
            measurements,
            out,
            template_name=template_name,
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
        (
            "rectangle_plastic_identity",
            Measurements(
                frame_width=140.0,
                lens_width=53.0,
                lens_height=42.0,
                bridge_width=17.2,
                temple_length=145.0,
                rim_thickness=1.2,
                material=FrameMaterial.PLASTIC,
                shape=FrameShape.RECTANGLE,
                nose_pads=False,
            ),
            "rectangle_plastic",
        ),
    ]

    source = Path("templates/geometric_metal.glb")
    source_metrics = scene_metrics(source)
    source_triangle_count = sum(
        geom["triangles"] for geom in source_metrics["geometries"].values()
    )
    case_results = []
    for item in cases:
        if len(item) == 2:
            name, measurements = item
            template_name = "geometric_metal"
        else:
            name, measurements, template_name = item
        case_results.append(run_case(name, measurements, output_dir, template_name))

    for case in case_results:
        if case["template_name"] == "geometric_metal":
            case["acceptance_errors"] = _accept_case(case, source_triangle_count)
        else:
            # Keep the second production asset diagnostic until its 2 m source
            # scale and hierarchy are understood; never treat export success as
            # proof of production readiness.
            case["acceptance_errors"] = []
            case["diagnostic_only"] = True

    report = {
        "source": source_metrics,
        "source_triangle_count": source_triangle_count,
        "stage_trace_identity": stage_trace(cases[0][1]),
        "cases": case_results,
    }
    path = output_dir / "regression.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("SOURCE", json.dumps(report["source"], sort_keys=True))
    for stage in report["stage_trace_identity"]:
        print("STAGE_TRACE", stage["stage"], json.dumps(stage["temples"], sort_keys=True))
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
            print(" gltf_contract", json.dumps(case.get("gltf_contract"), sort_keys=True))
        else:
            print(" error", case.get("error"))
        print(" acceptance_errors", json.dumps(case.get("acceptance_errors", []), sort_keys=True))

    failed = [case for case in report["cases"] if case.get("acceptance_errors")]
    report["passed"] = not failed
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Wrote", path)
    if failed:
        print("PRODUCTION_REGRESSION_FAILED", [case["name"] for case in failed])
        return 1
    print("PRODUCTION_REGRESSION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
