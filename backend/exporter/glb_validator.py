"""Production GLB validation for deformation outputs.

The validator intentionally checks the exported binary, not only the in-memory
scene, so a successful Trimesh export is never treated as proof of correctness.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from backend.models import Measurements


_REQUIRED_PARTS = (
    "Frame",
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftLens",
    "RightLens",
    "LeftTemple",
    "RightTemple",
)


@dataclass
class GLBValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB is too small to contain a valid header")
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("Invalid GLB magic")
    if version != 2:
        raise ValueError(f"Unsupported glTF binary version: {version}")
    if total_length != len(data):
        raise ValueError(
            f"GLB header length mismatch: header={total_length}, file={len(data)}"
        )

    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
    raise ValueError("GLB has no JSON chunk")


def _primitive_materials(gltf: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    materials = gltf.get("materials", [])
    result: dict[str, list[dict[str, Any]]] = {}
    for mesh in gltf.get("meshes", []):
        name = str(mesh.get("name") or "")
        result[name] = []
        for primitive in mesh.get("primitives", []):
            material_index = primitive.get("material")
            material = (
                materials[material_index]
                if isinstance(material_index, int) and 0 <= material_index < len(materials)
                else {}
            )
            result[name].append(material)
    return result


def validate_glb(
    path: str | Path,
    measurements: Measurements,
    *,
    frame_tolerance_mm: float = 5.0,
) -> GLBValidationReport:
    path = Path(path)
    report = GLBValidationReport()
    report.metrics["file_size_bytes"] = path.stat().st_size if path.exists() else 0

    if not path.exists() or path.stat().st_size == 0:
        report.fail("GLB file does not exist or is empty")
        return report

    try:
        gltf = _read_glb_json(path)
    except Exception as exc:
        report.fail(f"GLB structural parse failed: {exc}")
        return report

    asset = gltf.get("asset", {})
    report.metrics["gltf_version"] = asset.get("version")
    report.metrics["asset_generator"] = asset.get("generator")
    if asset.get("version") != "2.0":
        report.fail(f"Expected glTF 2.0, got {asset.get('version')!r}")

    scene_index = int(gltf.get("scene", 0))
    scenes = gltf.get("scenes", [])
    scene_json = scenes[scene_index] if 0 <= scene_index < len(scenes) else {}
    extras = scene_json.get("extras", {})
    if isinstance(extras, dict) and "extras" in extras:
        report.fail("Scene metadata contains unintended nested extras.extras")

    if isinstance(extras, dict):
        generator = extras.get("generator")
        if generator and str(generator).startswith("defirmation"):
            report.fail("Project generator metadata still contains the 'defirmation' typo")
        units = extras.get("units", {})
        if units:
            if units.get("geometry") != "meter":
                report.fail("GLB geometry units metadata must be 'meter'")
            if units.get("anchors") != "meter":
                report.fail("GLB anchor units metadata must be 'meter'")
            if units.get("measurements") != "millimeter":
                report.fail("Measurement units metadata must be 'millimeter'")
        else:
            report.fail("GLB is missing explicit units metadata")

        anchors = extras.get("anchors")
        if not isinstance(anchors, dict):
            report.fail("GLB is missing authoritative anchor metadata")
        else:
            for name in ("NoseBridge", "LeftHinge", "RightHinge"):
                value = anchors.get(name)
                if not (
                    isinstance(value, list)
                    and len(value) == 3
                    and all(np.isfinite(float(v)) for v in value)
                ):
                    report.fail(f"Invalid anchor: {name}")
            left = anchors.get("LeftHinge")
            right = anchors.get("RightHinge")
            if isinstance(left, list) and isinstance(right, list):
                if np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)) < 1e-5:
                    report.fail("Left and right hinge anchors collapse to the same point")

    # No second authoritative anchor block is allowed at the glTF root.
    root_extras = gltf.get("extras")
    if isinstance(root_extras, dict) and "anchors" in root_extras:
        report.fail("Duplicate authoritative anchors found in root extras")

    try:
        scene = trimesh.load(path, force="scene")
    except Exception as exc:
        report.fail(f"Trimesh failed to load exported GLB: {exc}")
        return report

    geometry = {
        name: mesh
        for name, mesh in scene.geometry.items()
        if isinstance(mesh, trimesh.Trimesh)
    }
    report.metrics["geometry_count"] = len(geometry)
    report.metrics["geometry_names"] = sorted(geometry)

    missing = [name for name in _REQUIRED_PARTS if name not in geometry]
    if missing:
        report.fail(f"Missing required independently addressable components: {missing}")

    if not missing:
        ids = [id(geometry[name]) for name in _REQUIRED_PARTS]
        if len(set(ids)) != len(ids):
            report.fail("Required components are aliased to the same mesh object")

    total_vertices = 0
    total_triangles = 0
    degenerate_triangles = 0
    unreferenced_vertices = 0

    for name, mesh in geometry.items():
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        total_vertices += len(vertices)
        total_triangles += len(faces)

        if not np.all(np.isfinite(vertices)):
            report.fail(f"{name} contains non-finite POSITION values")

        if len(faces):
            nondegenerate = mesh.nondegenerate_faces()
            degenerate_triangles += int((~nondegenerate).sum())
            referenced = np.unique(faces.reshape(-1))
            unreferenced_vertices += max(0, len(vertices) - len(referenced))

        normals = np.asarray(mesh.vertex_normals)
        if len(normals) != len(vertices) or not np.all(np.isfinite(normals)):
            report.fail(f"{name} has invalid vertex normals")
        elif len(normals) and np.any(np.linalg.norm(normals, axis=1) < 1e-8):
            report.fail(f"{name} has zero-length vertex normals")

    report.metrics.update(
        {
            "vertices": total_vertices,
            "triangles": total_triangles,
            "degenerate_triangles": degenerate_triangles,
            "unreferenced_vertices": unreferenced_vertices,
            "material_count": len(gltf.get("materials", [])),
            "draw_calls": sum(
                len(mesh.get("primitives", [])) for mesh in gltf.get("meshes", [])
            ),
        }
    )
    if degenerate_triangles:
        report.fail(f"Export contains {degenerate_triangles} degenerate triangles")
    if unreferenced_vertices:
        report.fail(f"Export contains {unreferenced_vertices} unreferenced vertices")

    if "Frame" in geometry:
        width_m = float(geometry["Frame"].bounds[1, 0] - geometry["Frame"].bounds[0, 0])
        expected_m = float(measurements.frame_width) / 1000.0
        error_mm = abs(width_m - expected_m) * 1000.0
        report.metrics["frame_width_m"] = width_m
        report.metrics["frame_width_error_mm"] = error_mm
        if error_mm > frame_tolerance_mm:
            report.fail(
                f"World-space frame width error {error_mm:.3f} mm exceeds "
                f"{frame_tolerance_mm:.3f} mm tolerance"
            )

    # Ensure normals are physically present in glTF, not merely reconstructed by Trimesh.
    missing_normal_primitives = 0
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "NORMAL" not in primitive.get("attributes", {}):
                missing_normal_primitives += 1
    report.metrics["primitives_missing_normal"] = missing_normal_primitives
    if missing_normal_primitives:
        report.fail(f"{missing_normal_primitives} primitives omit NORMAL attributes")

    # UVs are optional unless a material actually references textures.
    texture_required = bool(gltf.get("textures"))
    if texture_required:
        missing_uv = 0
        for mesh in gltf.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                if "TEXCOORD_0" not in primitive.get("attributes", {}):
                    missing_uv += 1
        if missing_uv:
            report.fail(f"{missing_uv} textured primitives omit TEXCOORD_0")

    material_map = _primitive_materials(gltf)
    for lens_name in ("LeftLens", "RightLens"):
        for material in material_map.get(lens_name, []):
            if material.get("alphaMode", "OPAQUE") == "OPAQUE":
                report.fail(f"{lens_name} uses an opaque lens material")

    return report
