"""Export deformed scene to GLB with validated units, metadata, and anchors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import trimesh

from backend.models import ExportMetadata, Measurements


ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]
MM_TO_METERS = 0.001
DEFAULT_FRAME_TOLERANCE_MM = float(os.environ.get("DEFIRM_EXPORT_FRAME_TOLERANCE_MM", "5.0"))


class GLBExporter:
    """Export a deformation-space (millimetre) scene as a glTF 2.0 GLB in metres."""

    def export(
        self,
        scene: trimesh.Scene,
        output_path: Path | str,
        measurements: Measurements,
        template_name: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._validate_source_dimensions(scene, measurements)

        export_scene = self._prepare_export_scene(scene)
        export_scene.apply_transform(
            np.array(
                [
                    [MM_TO_METERS, 0.0, 0.0, 0.0],
                    [0.0, MM_TO_METERS, 0.0, 0.0],
                    [0.0, 0.0, MM_TO_METERS, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

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
        anchors_m = self._compute_anchors_from_scene(export_scene, measurements, fallback_scale=MM_TO_METERS)
        self._attach_metadata(export_scene, metadata, anchors_m)

        export_scene.export(
            str(output_path),
            file_type="glb",
            include_normals=True,
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"GLB export failed: {output_path}")
        return output_path

    def _prepare_export_scene(self, scene: trimesh.Scene) -> trimesh.Scene:
        """Copy and safely clean geometry without changing deformation-critical topology upstream."""
        export_scene = scene.copy()
        for geom in export_scene.geometry.values():
            if not isinstance(geom, trimesh.Trimesh):
                continue

            # Only remove objectively invalid topology at the final export boundary.
            nondegenerate = geom.nondegenerate_faces()
            if len(nondegenerate) == len(geom.faces) and not np.all(nondegenerate):
                geom.update_faces(nondegenerate)
            geom.remove_unreferenced_vertices()

            # Recalculate finite outward-consistent normals after deformation/cleanup.
            geom.fix_normals()
            normals = np.asarray(geom.vertex_normals)
            if len(normals) != len(geom.vertices):
                raise ValueError("Failed to generate one normal per exported vertex")
            lengths = np.linalg.norm(normals, axis=1)
            if not np.all(np.isfinite(normals)) or np.any(lengths < 1e-8):
                raise ValueError("Export geometry contains invalid vertex normals")
            geom._cache.clear()

        return export_scene

    def _validate_source_dimensions(self, scene: trimesh.Scene, measurements: Measurements) -> None:
        """Validate deformation-space dimensions before the metre conversion."""
        frame = scene.geometry.get("Frame")
        if frame is None or not isinstance(frame, trimesh.Trimesh) or len(frame.vertices) == 0:
            raise ValueError("Cannot export without independently addressable Frame geometry")

        frame_width_mm = float(frame.bounds[1, 0] - frame.bounds[0, 0])
        error_mm = abs(frame_width_mm - float(measurements.frame_width))
        if error_mm > DEFAULT_FRAME_TOLERANCE_MM:
            raise ValueError(
                "Frame width regression: "
                f"geometry={frame_width_mm:.3f} mm, expected={measurements.frame_width:.3f} mm, "
                f"error={error_mm:.3f} mm, tolerance={DEFAULT_FRAME_TOLERANCE_MM:.3f} mm"
            )

    def _compute_anchors_from_scene(
        self,
        scene: trimesh.Scene,
        measurements: Measurements,
        *,
        fallback_scale: float = 1.0,
    ) -> dict[str, list[float]]:
        """Compute anchors in the same coordinate units as *scene*."""
        bridge = scene.geometry.get("Bridge")
        frame = scene.geometry.get("Frame")
        left_temple = scene.geometry.get("LeftTemple")
        right_temple = scene.geometry.get("RightTemple")

        reference = bridge if bridge is not None else frame
        if reference is not None and len(reference.vertices) > 0:
            nose = np.asarray(reference.vertices, dtype=np.float64).mean(axis=0)
        else:
            nose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        left_hinge = (
            left_temple.vertices[left_temple.vertices[:, 0].argmax()]
            if left_temple is not None and len(left_temple.vertices) > 0
            else np.array([-measurements.frame_width * 0.5 * fallback_scale, 0.0, 0.0])
        )
        right_hinge = (
            right_temple.vertices[right_temple.vertices[:, 0].argmin()]
            if right_temple is not None and len(right_temple.vertices) > 0
            else np.array([measurements.frame_width * 0.5 * fallback_scale, 0.0, 0.0])
        )

        return {
            "NoseBridge": np.asarray(nose, dtype=np.float64).tolist(),
            "LeftHinge": np.asarray(left_hinge, dtype=np.float64).tolist(),
            "RightHinge": np.asarray(right_hinge, dtype=np.float64).tolist(),
        }

    def compute_anchors_meters(
        self,
        scene_mm: trimesh.Scene,
        measurements: Measurements,
    ) -> dict[str, list[float]]:
        """Public helper for sidecar metadata; derives from the same anchor implementation."""
        anchors_mm = self._compute_anchors_from_scene(scene_mm, measurements)
        return {
            name: (np.asarray(value, dtype=np.float64) * MM_TO_METERS).tolist()
            for name, value in anchors_mm.items()
        }

    def _attach_metadata(
        self,
        scene: trimesh.Scene,
        metadata: ExportMetadata,
        anchors_m: dict[str, list[float]],
    ) -> None:
        # Trimesh serializes scene.metadata directly as glTF scene.extras.
        # Do not store another "extras" object here, which creates extras.extras.
        existing = dict(scene.metadata or {})
        legacy_extras = existing.pop("extras", None)
        if isinstance(legacy_extras, dict):
            for key, value in legacy_extras.items():
                if key not in {"defirmation", "deformation", "anchors", "units", "generator"}:
                    existing.setdefault(key, value)

        # Remove loader/runtime bookkeeping rather than exporting local paths.
        existing.pop("file_path", None)
        existing.pop("file_name", None)

        existing.update(
            {
                "generator": "deformation/2.0",
                "deformation": metadata.to_dict(),
                "anchors": anchors_m,
                "units": {
                    "geometry": "meter",
                    "anchors": "meter",
                    "measurements": "millimeter",
                },
            }
        )
        scene.metadata = existing

    def export_metadata_json(
        self,
        metadata: ExportMetadata,
        anchors: dict[str, list[float]],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        payload = {
            **metadata.to_dict(),
            "anchors": anchors,
            "units": {
                "geometry": "meter",
                "anchors": "meter",
                "measurements": "millimeter",
            },
            "generator": "deformation/2.0",
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path
