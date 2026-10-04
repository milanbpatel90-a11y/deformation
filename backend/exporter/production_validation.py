"""Fail-closed structural and dimensional checks for exported production GLBs.

The exporter remains deliberately unchanged. This module independently reads
the serialized GLB after export so unit/accessor/material mistakes cannot be
hidden by the in-memory scene that produced it.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from backend.exporter.validation import accessor, read_glb, world_primitives
from backend.models import Measurements


class ProductionValidationError(RuntimeError):
    """The serialized model violates a required production contract."""


def validate_production_glb(
    path: Path | str,
    measurements: Measurements,
    *,
    tolerance_mm: float = 0.5,
    check_self_intersections: bool | None = None,
    require_calibrated_scale: bool | None = None,
) -> dict:
    """Validate serialized attributes, scale, measured fit and optional BVH checks.

    Exact intersection checks are deliberately explicit because Open3D is not
    available on every supported Python runtime and can add substantial
    latency. A deployment that enables ``DEFIRM_PRODUCTION_MODE`` must request
    them; ordinary development output is marked REVIEW when they were omitted.
    """
    path = Path(path)
    report = {"status": "FAIL", "checks": {}, "errors": [], "warnings": []}
    production_mode = os.getenv("DEFIRM_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
    if require_calibrated_scale is None:
        require_calibrated_scale = production_mode
    report["checks"]["measurement_scale_source"] = measurements.measurement_scale_source
    report["checks"]["measurement_scale_calibrated"] = measurements.measurement_scale_calibrated
    report["checks"]["measurement_reference_width_mm"] = measurements.measurement_reference_width_mm
    if require_calibrated_scale and not measurements.measurement_scale_calibrated:
        report["errors"].append(
            "Production generation requires a known physical image scale reference; image-only size estimates are review-only"
        )
    tree, binary = read_glb(path)
    if tree.get("asset", {}).get("version") != "2.0":
        report["errors"].append("GLB asset version must be 2.0")
    report["checks"]["gltf_version"] = tree.get("asset", {}).get("version")
    scenes = tree.get("scenes", [])
    active_scene = tree.get("scene", 0)
    if not isinstance(active_scene, int) or active_scene < 0 or active_scene >= len(scenes):
        report["errors"].append("GLB references a missing active scene")
        scene_extras = {}
    else:
        scene_extras = scenes[active_scene].get("extras", {})
    if scene_extras.get("coordinate_units") != "m":
        report["errors"].append("GLB scene metadata must declare coordinate_units=m")
    if scene_extras.get("measurement_units") != "mm":
        report["errors"].append("GLB scene metadata must declare measurement_units=mm")
    buffers = tree.get("buffers", [])
    if len(buffers) != 1 or "uri" in (buffers[0] if buffers else {}):
        report["errors"].append("Production GLB must contain one embedded binary buffer")
    elif buffers[0].get("byteLength", len(binary)) > len(binary):
        report["errors"].append("GLB binary chunk is shorter than its declared buffer")
    report["checks"]["coordinate_units"] = scene_extras.get("coordinate_units")
    report["checks"]["measurement_units"] = scene_extras.get("measurement_units")

    material_textures = set()
    for material in tree.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        for block in (pbr, material):
            for key in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                        "occlusionTexture", "emissiveTexture"):
                texture = block.get(key)
                if isinstance(texture, dict):
                    material_textures.add(material.get("name", ""))

    grouped: dict[str, list[np.ndarray]] = {}
    triangle_count = 0
    primitive_count = 0
    uv_required = bool(material_textures)
    try:
        for node, primitive, points in world_primitives(tree, binary):
            primitive_count += 1
            name = node.get("name", "")
            grouped.setdefault(name, []).append(points)
            attrs = primitive.get("attributes", {})
            positions = accessor(tree, binary, attrs["POSITION"])
            normals = accessor(tree, binary, attrs["NORMAL"]) if "NORMAL" in attrs else None
            if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
                report["errors"].append(f"{name}: POSITION must be finite VEC3")
                continue
            if normals is None or normals.shape != positions.shape or not np.isfinite(normals).all():
                report["errors"].append(f"{name}: valid vertex NORMAL data is required")
            elif not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3):
                report["errors"].append(f"{name}: normals must be normalized")
            if uv_required and primitive.get("material") is not None:
                mat_name = tree["materials"][primitive["material"]].get("name", "")
                if mat_name in material_textures and "TEXCOORD_0" not in attrs:
                    report["errors"].append(f"{name}: textured material requires TEXCOORD_0")
            if "TEXCOORD_0" in attrs:
                uv = accessor(tree, binary, attrs["TEXCOORD_0"])
                if uv.shape != (len(positions), 2) or not np.isfinite(uv).all():
                    report["errors"].append(f"{name}: TEXCOORD_0 must be finite VEC2 per vertex")

            indices = accessor(tree, binary, primitive["indices"]).reshape(-1)
            if len(indices) % 3 or not len(indices) or int(indices.max()) >= len(positions):
                report["errors"].append(f"{name}: triangle indices are empty, incomplete or out of range")
                continue
            triangles = positions[indices.astype(np.int64)].reshape(-1, 3, 3)
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                             triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
            if not np.isfinite(areas).all() or np.any(areas <= 1e-14):
                report["errors"].append(f"{name}: degenerate or non-finite triangles found")
            triangle_count += len(indices) // 3
    except (KeyError, IndexError, TypeError, ValueError, struct.error) as exc:
        report["errors"].append(f"Malformed GLB data: {exc}")

    report["checks"].update({"primitive_count": primitive_count, "triangle_count": triangle_count,
                              "normals_required": True, "uv_required_by_material": uv_required})
    extents = {}
    bounds = {}
    for name, instances in grouped.items():
        vertices = np.vstack(instances)
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        extents[name] = (maximum - minimum).tolist()
        bounds[name] = [minimum.tolist(), maximum.tolist()]
        if not np.isfinite(vertices).all() or np.any(np.asarray(extents[name]) <= 0):
            report["errors"].append(f"{name}: invalid or zero-volume bounds")
        if np.max(np.abs(vertices)) > 1.0:
            report["errors"].append(f"{name}: world coordinates exceed the 1 metre eyewear envelope")

    required_parts = {"Frame", "LeftLens", "RightLens"}
    missing = sorted(required_parts - grouped.keys())
    if missing:
        report["errors"].append(f"Missing required GLB components: {', '.join(missing)}")
    elif isinstance(scene_extras.get("deformation_validation"), dict):
        frame_width = extents["Frame"][0] * 1000.0
        left, right = bounds["LeftLens"], bounds["RightLens"]
        left_width = extents["LeftLens"][0] * 1000.0
        right_width = extents["RightLens"][0] * 1000.0
        # Exported eyewear is Y-up; width is X and lens height is Y.
        left_height = extents["LeftLens"][1] * 1000.0
        right_height = extents["RightLens"][1] * 1000.0
        bridge_gap = (left[0][0] - right[1][0]) * 1000.0
        measured = {"frame_width": frame_width, "left_lens_width": left_width,
                    "right_lens_width": right_width, "left_lens_height": left_height,
                    "right_lens_height": right_height, "bridge_width": bridge_gap}
        expected = {"frame_width": measurements.frame_width, "left_lens_width": measurements.lens_width,
                    "right_lens_width": measurements.lens_width, "left_lens_height": measurements.lens_height,
                    "right_lens_height": measurements.lens_height, "bridge_width": measurements.bridge_width}
        errors = {key: abs(measured[key] - target) for key, target in expected.items()}
        report["checks"]["measured_mm"] = measured
        report["checks"]["dimension_errors_mm"] = errors
        report["checks"]["maximum_dimension_error_mm"] = max(errors.values())
        if max(errors.values()) > tolerance_mm:
            report["errors"].append(
                f"Serialized GLB dimensional error {max(errors.values()):.4g} mm exceeds {tolerance_mm:g} mm"
            )
    else:
        # Legacy procedural templates do not carry a calibrated deformation
        # certificate, so their apparent bounds cannot prove dimensional fit.
        # Keep their development exports inspectable, while production mode
        # rejects REVIEW until they are migrated to a calibrated basis bundle.
        report["checks"]["dimensional_validation"] = "not_applicable_uncalibrated_template"
        report["warnings"].append(
            "Template has no calibrated deformation certificate; physical dimensions were not verified"
        )

    run_intersections = (production_mode if check_self_intersections is None else check_self_intersections)
    if run_intersections:
        try:
            import open3d as o3d
        except ImportError as exc:
            report["errors"].append("Exact self-intersection check requested but Open3D is unavailable")
            report["checks"]["self_intersections"] = {"status": "unavailable"}
        else:
            findings = {}
            for name, instances in grouped.items():
                # Current templates have one node instance per part. Combine
                # node instances in world space without changing export data.
                if len(instances) != 1:
                    report["errors"].append(f"{name}: multiple instances require a dedicated collision audit")
                    continue
                # Read that node's index buffer to preserve the exported faces.
                faces = None
                for node, primitive, _ in world_primitives(tree, binary):
                    if node.get("name") == name:
                        faces = accessor(tree, binary, primitive["indices"]).reshape(-1, 3).astype(np.int32)
                        break
                if faces is None:
                    report["errors"].append(f"{name}: could not resolve triangles for intersection check")
                    continue
                # GLB POSITION values are commonly float32, whose quantization
                # can separate a shared seam by a few micrometres. Weld at
                # 0.0001 mm in metre coordinates for collision diagnostics only.
                welded_vertices, inverse = np.unique(np.round(instances[0], decimals=7),
                                                       axis=0, return_inverse=True)
                faces = inverse[faces]
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(welded_vertices),
                    o3d.utility.Vector3iVector(faces),
                )
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                indices = np.asarray(mesh.get_self_intersecting_triangles())
                findings[name] = int(len(indices))
            report["checks"]["self_intersections"] = {"status": "checked", "triangles_by_part": findings}
            if any(findings.values()):
                report["errors"].append(f"Self-intersecting triangles detected: {findings}")
    else:
        report["checks"]["self_intersections"] = {"status": "not_checked"}
        report["warnings"].append("Exact triangle self-intersection validation was not run")

    report["status"] = "FAIL" if report["errors"] else (
        "REVIEW" if report["warnings"] else "PASS")
    if report["status"] == "FAIL":
        raise ProductionValidationError("; ".join(report["errors"]))
    return report
