"""Export deformed scene to GLB with metadata and anchors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from backend.models import ExportMetadata, Measurements


# Standard VTO anchor points (nose bridge, temple hinges)
ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]


class GLBExporter:
    """Export trimesh scene to GLB with embedded metadata."""

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
        )

        export_scene = scene.copy()
        self._remove_alias_geometry(export_scene)
        anchors = self._compute_anchors(export_scene, measurements)
        self._attach_metadata(export_scene, metadata, anchors)

        export_scene.export(str(output_path), file_type="glb")
        return output_path

    @staticmethod
    def _remove_alias_geometry(scene: trimesh.Scene) -> None:
        """Remove logical alias entries before serializing the scene.

        Deformation contexts may expose aliases such as ``LeftRim`` that point
        at the same mesh as the original GLB entry. Serializing those aliases
        creates visible duplicate geometry in the exported model.
        """
        raw_aliases = scene.metadata.get("mesh_aliases", {}) if scene.metadata else {}
        if not isinstance(raw_aliases, dict):
            return
        for logical_name, actual_name in raw_aliases.items():
            if logical_name != actual_name and logical_name in scene.geometry:
                scene.delete_geometry(logical_name)

    def _compute_anchors(
        self, scene: trimesh.Scene, measurements: Measurements
    ) -> dict[str, list[float]]:
        """Compute VTO attachment points from deformed geometry."""
        bridge = scene.geometry.get("Bridge") or scene.geometry.get("Frame")
        left_temple = scene.geometry.get("LeftTemple")
        right_temple = scene.geometry.get("RightTemple")

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
            "NoseBridge": nose.tolist(),
            "LeftHinge": left_hinge.tolist(),
            "RightHinge": right_hinge.tolist(),
        }

    def _attach_metadata(
        self,
        scene: trimesh.Scene,
        metadata: ExportMetadata,
        anchors: dict[str, list[float]],
    ) -> None:
        payload = {
            "defirmation": metadata.to_dict(),
            "anchors": anchors,
        }
        scene.metadata = scene.metadata or {}
        scene.metadata["extras"] = payload

        # Also write sidecar JSON for non-GLB consumers
        sidecar = Path(str(metadata.template_used))  # placeholder; set by caller if needed
        scene.metadata["generator"] = "defirmation/1.0"

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
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path
