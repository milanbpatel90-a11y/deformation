"""Validated GLB export for the eyewear deformation pipeline.

All internal geometry and measurements use millimetres. glTF/GLB geometry uses
metres, so the scene is converted at the export boundary and anchor coordinates
are written in metres. This keeps the deformer in one consistent unit system.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh

from backend.models import ExportMetadata, Measurements


ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]
MM_TO_M = 0.001


class GLBExporter:
    """Validate, convert, and export a deformed scene as a standard-unit GLB."""

    @staticmethod
    def _validate_scene(scene: trimesh.Scene, stage: str) -> None:
        meshes = [
            geom for geom in scene.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"{stage}: scene contains no triangle meshes.")

        total_vertices = 0
        total_faces = 0
        for name, mesh in scene.geometry.items():
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
                raise ValueError(f"{stage}: mesh {name!r} has invalid or empty vertices.")
            if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
                raise ValueError(f"{stage}: mesh {name!r} has no valid triangle faces.")
            if not np.isfinite(vertices).all():
                raise ValueError(f"{stage}: mesh {name!r} contains NaN or infinite vertices.")
            if faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError(f"{stage}: mesh {name!r} contains out-of-range face indices.")
            total_vertices += len(vertices)
            total_faces += len(faces)

        if total_vertices < 3 or total_faces < 1:
            raise ValueError(f"{stage}: no usable triangle geometry was found.")

    @staticmethod
    def _compute_anchors(
        scene: trimesh.Scene,
        measurements: Measurements,
        coordinate_scale: float = 1.0,
    ) -> dict[str, list[float]]:
        """Compute VTO anchors, optionally converting their coordinate units."""
        def world_vertices(name: str) -> np.ndarray | None:
            mesh = scene.geometry.get(name)
            if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                return None
            transformed = []
            for node_name in scene.graph.nodes_geometry:
                matrix, geometry_name = scene.graph[node_name]
                if geometry_name != name:
                    continue
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                transformed.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
            # DeformationContext may inject independently copied logical meshes
            # directly into scene.geometry; those are already baked into world space.
            return np.concatenate(transformed, axis=0) if transformed else np.asarray(mesh.vertices)

        bridge_vertices = world_vertices("Bridge")
        if bridge_vertices is None:
            bridge_vertices = world_vertices("Frame")
        left_vertices = world_vertices("LeftTemple")
        right_vertices = world_vertices("RightTemple")

        nose = bridge_vertices.mean(axis=0) if bridge_vertices is not None else np.zeros(3)
        left_hinge = (
            left_vertices[np.argmin(left_vertices[:, 0])]
            if left_vertices is not None
            else np.array([-measurements.frame_width / 2.0, 0.0, 0.0])
        )
        right_hinge = (
            right_vertices[np.argmax(right_vertices[:, 0])]
            if right_vertices is not None
            else np.array([measurements.frame_width / 2.0, 0.0, 0.0])
        )
        anchors = {
            "NoseBridge": np.asarray(nose, dtype=np.float64),
            "LeftHinge": np.asarray(left_hinge, dtype=np.float64),
            "RightHinge": np.asarray(right_hinge, dtype=np.float64),
        }
        scaled = {}
        for name, point in anchors.items():
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"Cannot export invalid {name} anchor.")
            scaled[name] = (point * float(coordinate_scale)).tolist()
        return scaled

    def export(
        self,
        scene: trimesh.Scene,
        output_path: Path | str,
        measurements: Measurements,
        template_name: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not math.isfinite(float(measurements.frame_width)) or not 80.0 <= float(measurements.frame_width) <= 200.0:
            raise ValueError("Refusing to export: frame_width must be finite and within 80–200 mm.")

        self._validate_scene(scene, "pre-export")
        export_scene = deepcopy(scene)
        # Compute world-space anchors from the internal mm scene, then convert
        # them to metres. This also scales fallback anchors when a part is absent.
        anchors = self._compute_anchors(export_scene, measurements, coordinate_scale=MM_TO_M)

        # The internal pipeline is in mm. glTF geometry is in metres.
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= MM_TO_M
        export_scene.apply_transform(transform)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
        )
        payload = {
            "defirmation": metadata.to_dict(),
            "anchors": anchors,
            "units": {"geometry": "m", "measurements": "mm", "anchors": "m"},
            "template": template_name,
        }
        export_scene.metadata = dict(export_scene.metadata or {})
        export_scene.metadata["extras"] = payload
        export_scene.metadata["generator"] = "defirmation/1.0"

        temporary_path = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.glb")
        try:
            export_scene.export(str(temporary_path), file_type="glb")
            if not temporary_path.is_file() or temporary_path.stat().st_size < 512:
                raise ValueError("GLB export produced an empty or truncated file.")

            reloaded = trimesh.load(str(temporary_path), force="scene")
            if not isinstance(reloaded, trimesh.Scene):
                raise ValueError("Exported GLB could not be loaded as a scene.")
            self._validate_scene(reloaded, "post-export")

            bounds = np.asarray(reloaded.bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise ValueError("Exported GLB has invalid scene bounds.")
            extents = bounds[1] - bounds[0]
            max_extent_m = float(np.max(extents))
            if not 0.015 <= max_extent_m <= 0.6:
                raise ValueError(
                    f"Exported GLB has suspicious scale ({max_extent_m:.6f} m). "
                    "Expected a glasses-sized model in metre units."
                )

            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        return output_path

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
            "anchors": anchors,
            "units": {"geometry": "m", "measurements": "mm", "anchors": "m"},
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path
