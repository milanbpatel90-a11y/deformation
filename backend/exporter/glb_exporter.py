"""Export deformed scenes to standards-compliant GLB with metadata and anchors."""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh

from backend.models import ExportMetadata, Measurements


ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]
MM_TO_METERS = 0.001


class GLBExporter:
    """Export a validated millimetre-space scene to a metre-space glTF binary."""

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

        # Work on an isolated copy: never mutate the runtime template or context.
        export_scene = deepcopy(scene)
        anchors_m = self._anchors_to_meters(
            self._compute_anchors(export_scene, measurements)
        )
        self._attach_metadata(export_scene, metadata, anchors_m)
        self._remove_unreferenced_geometry(export_scene)
        self._scale_scene_to_meters(export_scene)

        export_scene.export(str(output_path), file_type="glb")
        self._validate_glb(output_path)
        return output_path

    @staticmethod
    def _world_mesh(scene: trimesh.Scene, name: str) -> trimesh.Trimesh | None:
        mesh = scene.geometry.get(name)
        if not isinstance(mesh, trimesh.Trimesh):
            return None

        # Alias-only logical meshes are already baked into world coordinates.
        matching_nodes = [
            node_name
            for node_name in scene.graph.nodes_geometry
            if scene.graph[node_name][1] == name
        ]
        if not matching_nodes:
            return mesh

        transform, _ = scene.graph[matching_nodes[0]]
        world_mesh = mesh.copy()
        if not np.allclose(transform, np.eye(4)):
            world_mesh.apply_transform(transform)
        return world_mesh

    def _compute_anchors(
        self, scene: trimesh.Scene, measurements: Measurements
    ) -> dict[str, list[float]]:
        """Compute VTO attachment points in world-space millimetres."""
        bridge = self._world_mesh(scene, "Bridge")
        if bridge is None:
            bridge = self._world_mesh(scene, "Frame")
        left_temple = self._world_mesh(scene, "LeftTemple")
        right_temple = self._world_mesh(scene, "RightTemple")

        if bridge is not None and len(bridge.vertices) > 0:
            nose = bridge.vertices.mean(axis=0)
        else:
            nose = np.array([0.0, 0.0, 0.0])

        left_hinge = (
            left_temple.vertices[left_temple.vertices[:, 0].argmin()]
            if left_temple is not None and len(left_temple.vertices) > 0
            else np.array([-measurements.frame_width / 2, 0.0, 0.0])
        )
        right_hinge = (
            right_temple.vertices[right_temple.vertices[:, 0].argmax()]
            if right_temple is not None and len(right_temple.vertices) > 0
            else np.array([measurements.frame_width / 2, 0.0, 0.0])
        )

        return {
            "NoseBridge": np.asarray(nose, dtype=float).tolist(),
            "LeftHinge": np.asarray(left_hinge, dtype=float).tolist(),
            "RightHinge": np.asarray(right_hinge, dtype=float).tolist(),
        }

    @staticmethod
    def _anchors_to_meters(
        anchors_mm: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        return {
            name: (np.asarray(point, dtype=np.float64) * MM_TO_METERS).tolist()
            for name, point in anchors_mm.items()
        }

    @staticmethod
    def _remove_unreferenced_geometry(scene: trimesh.Scene) -> None:
        """Drop alias-only copies so GLB export cannot duplicate rendered parts."""
        referenced = {
            scene.graph[node_name][1]
            for node_name in scene.graph.nodes_geometry
        }
        for geometry_name in list(scene.geometry):
            if geometry_name not in referenced:
                scene.geometry.pop(geometry_name, None)

    @staticmethod
    def _scale_scene_to_meters(scene: trimesh.Scene) -> None:
        """Convert local millimetre vertices and node translations to metres."""
        seen_geometry: set[int] = set()
        for geometry in scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh) or id(geometry) in seen_geometry:
                continue
            seen_geometry.add(id(geometry))
            geometry.vertices = np.asarray(geometry.vertices, dtype=np.float64) * MM_TO_METERS

        for node_name in scene.graph.nodes_geometry:
            transform, _ = scene.graph[node_name]
            scaled = np.asarray(transform, dtype=np.float64).copy()
            scaled[:3, 3] *= MM_TO_METERS
            scene.graph.update(frame_to=node_name, matrix=scaled)

    @staticmethod
    def _attach_metadata(
        scene: trimesh.Scene,
        metadata: ExportMetadata,
        anchors_m: dict[str, list[float]],
    ) -> None:
        scene.metadata = scene.metadata or {}
        scene.metadata["extras"] = {
            "defirmation": metadata.to_dict(),
            "anchors": anchors_m,
            "units": {
                "geometry": "m",
                "anchors": "m",
                "measurements": "mm",
            },
        }
        scene.metadata["generator"] = "defirmation/2.0"

    @staticmethod
    def _validate_glb(output_path: Path) -> None:
        """Verify the GLB header, declared length, and reloaded mesh buffers."""
        if not output_path.is_file():
            raise RuntimeError("GLB export did not create an output file.")

        file_size = output_path.stat().st_size
        with output_path.open("rb") as handle:
            header = handle.read(12)
        if len(header) != 12:
            raise RuntimeError("GLB export is truncated.")
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or declared_length != file_size:
            raise RuntimeError("GLB export failed glTF 2.0 header/length validation.")

        try:
            reloaded = trimesh.load(str(output_path), force="scene")
        except Exception as exc:
            raise RuntimeError("Exported GLB could not be reloaded.") from exc

        valid_meshes = 0
        for name, mesh in reloaded.geometry.items():
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            if not np.isfinite(vertices).all():
                raise RuntimeError(f"Exported GLB mesh '{name}' contains non-finite vertices.")
            if len(vertices) >= 3 and len(faces) > 0:
                valid_meshes += 1
        if valid_meshes == 0:
            raise RuntimeError("Exported GLB contains no valid triangle meshes.")

    def export_metadata_json(
        self,
        metadata: ExportMetadata,
        anchors: dict[str, list[float]],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata.to_dict(),
            "anchors": self._anchors_to_meters(anchors),
            "units": {
                "geometry": "m",
                "anchors": "m",
                "measurements": "mm",
            },
            "generator": "defirmation/2.0",
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path
