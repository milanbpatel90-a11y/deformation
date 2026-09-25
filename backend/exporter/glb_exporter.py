"""Export deformed scene to GLB with canonical metadata and VTO anchors."""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from backend.geometry_units import mm_to_m
from backend.models import ExportMetadata, Measurements

ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]
_GLTF_JSON = 0x4E4F534A
_GLTF_BIN = 0x004E4942


class GLBExporter:
    """Export a deformation runtime scene as a glTF 2.0 GLB."""

    def export(
        self,
        scene: trimesh.Scene,
        output_path: Path | str,
        measurements: Measurements,
        template_name: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
            lens_color=measurements.lens_color,
            lens_opacity=measurements.lens_opacity,
        )

        anchors = self._compute_anchors(scene, measurements)
        payload = self._canonical_payload(metadata, anchors)

        export_scene = self._prepare_export_scene(scene)
        exported = export_scene.export(file_type="glb")
        if not isinstance(exported, (bytes, bytearray)):
            raise RuntimeError("trimesh did not return GLB bytes")

        output_path.write_bytes(bytes(exported))
        self._rewrite_glb_metadata(output_path, payload)
        return output_path

    def _prepare_export_scene(self, scene: trimesh.Scene) -> trimesh.Scene:
        """Remove runtime-only proxy geometry while preserving real components."""
        export_scene = deepcopy(scene)
        runtime = export_scene.metadata.get("deformation_runtime", {})
        frame_proxy = runtime.get("frame_proxy") if isinstance(runtime, dict) else None

        if frame_proxy and frame_proxy in export_scene.geometry:
            export_scene.delete_geometry(frame_proxy)

        # Runtime-only bookkeeping must not leak into glTF scene extras.
        export_scene.metadata = {
            key: value
            for key, value in (export_scene.metadata or {}).items()
            if key != "deformation_runtime"
        }

        required = {
            "Bridge",
            "LeftRim",
            "RightRim",
            "LeftLens",
            "RightLens",
            "LeftTemple",
            "RightTemple",
        }
        missing = required.difference(export_scene.geometry)
        if missing:
            raise ValueError(
                "Export scene is missing independently addressable components: "
                + ", ".join(sorted(missing))
            )
        if export_scene.geometry["LeftLens"] is export_scene.geometry["RightLens"]:
            raise ValueError("Left and right lenses must be independent mesh objects")
        if export_scene.geometry["LeftTemple"] is export_scene.geometry["RightTemple"]:
            raise ValueError("Left and right temples must be independent mesh objects")

        return export_scene

    def _compute_anchors(
        self, scene: trimesh.Scene, measurements: Measurements
    ) -> dict[str, list[float]]:
        """Compute authoritative world-space meter anchors from deformed geometry."""
        bridge = scene.geometry.get("Bridge")
        frame = scene.geometry.get("Frame")
        left_temple = scene.geometry.get("LeftTemple")
        right_temple = scene.geometry.get("RightTemple")

        if bridge is not None and len(bridge.vertices) > 0:
            nose = np.asarray(bridge.vertices, dtype=np.float64).mean(axis=0)
        else:
            nose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        def hinge(temple: trimesh.Trimesh | None, fallback_x_mm: float) -> np.ndarray:
            if temple is not None and len(temple.vertices) > 0 and frame is not None and len(frame.vertices) > 0:
                tree = cKDTree(np.asarray(frame.vertices, dtype=np.float64))
                distances, _ = tree.query(np.asarray(temple.vertices, dtype=np.float64), k=1)
                return np.asarray(temple.vertices, dtype=np.float64)[int(np.argmin(distances))]
            return np.array([mm_to_m(fallback_x_mm), 0.0, 0.0], dtype=np.float64)

        left_hinge = hinge(left_temple, -measurements.frame_width / 2.0)
        right_hinge = hinge(right_temple, measurements.frame_width / 2.0)

        return {
            "NoseBridge": nose.tolist(),
            "LeftHinge": left_hinge.tolist(),
            "RightHinge": right_hinge.tolist(),
        }

    @staticmethod
    def _canonical_payload(
        metadata: ExportMetadata,
        anchors: dict[str, list[float]],
    ) -> dict[str, Any]:
        return {
            "deformation": metadata.to_dict(),
            "anchors": anchors,
            "units": {
                "geometry": "meter",
                "anchors": "meter",
                "measurements": "millimeter",
            },
        }

    @staticmethod
    def _rewrite_glb_metadata(path: Path, payload: dict[str, Any]) -> None:
        """Patch only the GLB JSON chunk; preserve binary geometry bytes exactly."""
        raw = path.read_bytes()
        if len(raw) < 20 or raw[:4] != b"glTF":
            raise ValueError("Exported file is not a GLB")

        version, declared_length = struct.unpack_from("<II", raw, 4)
        if version != 2 or declared_length != len(raw):
            raise ValueError("Exported GLB header is invalid")

        offset = 12
        chunks: list[tuple[int, bytes]] = []
        while offset + 8 <= len(raw):
            length, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            chunk = raw[offset : offset + length]
            offset += length
            chunks.append((chunk_type, chunk))

        json_index = next(
            (i for i, (chunk_type, _) in enumerate(chunks) if chunk_type == _GLTF_JSON),
            None,
        )
        if json_index is None:
            raise ValueError("Exported GLB has no JSON chunk")

        doc = json.loads(chunks[json_index][1].rstrip(b" \t\r\n\x00").decode("utf-8"))
        asset = doc.setdefault("asset", {})
        asset["version"] = "2.0"
        asset["generator"] = "deformation/1.0"

        scenes = doc.setdefault("scenes", [{"nodes": []}])
        scene_index = int(doc.get("scene", 0))
        while len(scenes) <= scene_index:
            scenes.append({"nodes": []})

        existing_extras = scenes[scene_index].get("extras")
        extras = dict(existing_extras) if isinstance(existing_extras, dict) else {}
        # Remove only the historical project nesting if it exists.
        legacy_nested = extras.pop("extras", None)
        if isinstance(legacy_nested, dict):
            for key, value in legacy_nested.items():
                extras.setdefault(key, value)
        extras.pop("defirmation", None)
        extras.pop("generator", None)
        extras.update(payload)
        scenes[scene_index]["extras"] = extras

        encoded = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * ((4 - len(encoded) % 4) % 4)
        chunks[json_index] = (_GLTF_JSON, encoded)

        body = bytearray()
        for chunk_type, chunk in chunks:
            padded = chunk + (b"\x00" * ((4 - len(chunk) % 4) % 4) if chunk_type != _GLTF_JSON else b"")
            body += struct.pack("<II", len(padded), chunk_type)
            body += padded

        header = b"glTF" + struct.pack("<II", 2, 12 + len(body))
        path.write_bytes(header + body)

    def export_metadata_json(
        self,
        metadata: ExportMetadata,
        anchors: dict[str, list[float]],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        payload = self._canonical_payload(metadata, anchors)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path
