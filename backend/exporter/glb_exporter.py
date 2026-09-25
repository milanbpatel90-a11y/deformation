"""Export millimetre deformation scenes as metre-based glTF 2.0 assets."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import trimesh
from backend.models import ExportMetadata, Measurements
from backend.exporter.geometry import dimensions, prepare_geometry, world_vertices

ANCHOR_NAMES = ["NoseBridge", "LeftHinge", "RightHinge"]


class GLBExporter:
    def export(self, scene: trimesh.Scene, output_path: Path | str,
               measurements: Measurements, template_name: str, *, source_units: str | None = None) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not scene.geometry or any(not isinstance(m, trimesh.Trimesh) or not len(m.vertices)
                                    or not np.isfinite(m.vertices).all() for m in scene.geometry.values()):
            raise ValueError("Cannot export empty or non-finite geometry")
        # Existing deformation API is mm. Re-exporting our metre GLB is idempotent.
        units = source_units or scene.metadata.get("coordinate_units", "mm")
        if units not in {"mm", "m"}:
            raise ValueError(f"Unsupported source coordinate units: {units}")
        factor = 0.001 if units == "mm" else 1.0
        exported = scene.copy()
        anchors = self._compute_anchors(exported, measurements)
        stats = prepare_geometry(exported)
        # Scale both local geometry and local translations: S*T*S^-1. This
        # retains parent-relative rotations and works for nested node transforms.
        for mesh in exported.geometry.values():
            mesh.apply_scale(factor)
            mesh.units = "m"
        for parent, child, data in list(exported.graph.to_edgelist()):
            data = deepcopy(data)
            matrix = np.asarray(data.pop("matrix"), dtype=float)
            matrix[:3, 3] *= factor
            exported.graph.update(frame_from=parent, frame_to=child, matrix=matrix, **data)
        anchors = {name: (np.asarray(point) * factor).tolist() for name, point in anchors.items()}
        self._attach_pivots(exported, anchors)
        metadata = ExportMetadata(shape=measurements.shape.value, material=measurements.material.value,
            frame_width=measurements.frame_width, bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length, template_used=template_name,
            color=measurements.color, lens_color=measurements.lens_color, lens_opacity=measurements.lens_opacity)
        self._attach_metadata(exported, metadata, anchors)
        exported.metadata["geometry_measurements"] = dimensions(exported)
        exported.metadata["export_cleanup"] = stats
        def finalize(tree):
            tree["asset"]["generator"] = "deformation/1.1 (trimesh)"
            for mesh in tree["meshes"]:
                for primitive in mesh["primitives"]:
                    for index in primitive["attributes"].values():
                        tree["bufferViews"][tree["accessors"][index]["bufferView"]]["target"] = 34962
                    tree["bufferViews"][tree["accessors"][primitive["indices"]]["bufferView"]]["target"] = 34963
        def compact_indices(buffers, tree):
            accessors = list(tree["accessors"].values())
            keys = list(buffers)
            indices = {p["indices"] for mesh in tree["meshes"] for p in mesh["primitives"]}
            for index in indices:
                item = accessors[index]
                if item["componentType"] != 5125 or item.get("byteOffset", 0):
                    continue
                key = keys[item["bufferView"]]
                values = np.frombuffer(buffers[key], dtype="<u4", count=item["count"])
                if values.max() < 65535:
                    packed = values.astype("<u2").tobytes()
                    buffers[key] = packed + b"\x00" * (-len(packed) % 4)
                    item["componentType"] = 5123
        payload = trimesh.exchange.gltf.export_glb(exported, include_normals=True,
            tree_postprocessor=finalize, buffer_postprocessor=compact_indices)
        output_path.write_bytes(payload)
        # Preserve the sidecar's public flat measurement keys in millimetres.
        # Anchors now explicitly share the GLB's world-space metre convention.
        self.export_metadata_json(metadata, anchors, output_path.with_suffix(".metadata.json"), anchor_units="m",
                                  measured=exported.metadata["geometry_measurements"])
        return output_path

    def _compute_anchors(self, scene, measurements):
        # Exported anchors are authoritative on re-import. Internal vto_anchors
        # are explicitly world-space in the source scene's coordinate units.
        existing = scene.metadata.get("anchors", scene.metadata.get("vto_anchors"))
        if existing:
            return deepcopy(existing)
        result = {"NoseBridge": [0.0, 0.0, 0.0]}
        if "Bridge" in scene.geometry:
            result["NoseBridge"] = world_vertices(scene, "Bridge").mean(0).tolist()
        for side, sign in (("Left", 1), ("Right", -1)):
            name = side + "Hinge"
            if name in scene.geometry:
                points = world_vertices(scene, name)
                result[name] = ((points.min(0) + points.max(0)) / 2).tolist()
            else:
                result[name] = [sign * measurements.frame_width / 2, 0.0, 0.0]
        return result

    @staticmethod
    def _attach_pivots(scene, anchors):
        # Rebuild graph edges to reparent without stale edges or duplicate parents.
        edges = deepcopy(scene.graph.to_edgelist())
        root = scene.graph.base_frame
        moving = {}
        for side in ("Left", "Right"):
            pivot = side + "TemplePivot"
            if pivot in scene.graph.nodes:
                continue  # Our exported hierarchy is already pivoted.
            transform = trimesh.transformations.translation_matrix(anchors[side + "Hinge"])
            edges.append((root, pivot, {"matrix": transform, "metadata": {
                "rotation_axis": [0, 1, 0], "anchor_ref": side + "Hinge"}}))
            for node in scene.graph.nodes_geometry:
                world, geometry = scene.graph[node]
                if geometry in {side + "Temple", side + "FrameInsert"}:
                    moving[node] = (pivot, np.linalg.inv(transform) @ world)
        graph = trimesh.scene.transforms.SceneGraph(base_frame=root)
        for parent, child, data in edges:
            if child in moving:
                parent, data["matrix"] = moving[child]
            graph.update(frame_from=parent, frame_to=child, **data)
        scene.graph = graph

    def _attach_metadata(self, scene, metadata, anchors):
        scene.metadata = deepcopy(scene.metadata or {})
        old = scene.metadata.get("extras")
        # Flatten only our legacy wrapper, preserving unrelated custom extras.
        if isinstance(old, dict) and "defirmation" in old:
            scene.metadata.pop("extras")
            for key, value in old.items():
                if key not in {"defirmation", "anchors"}:
                    scene.metadata.setdefault(key, value)
        scene.metadata.pop("vto_anchors", None)
        if str(scene.metadata.get("generator", "")).startswith("defirmation/"):
            scene.metadata.pop("generator")
        scene.metadata.pop("defirmation", None)
        scene.metadata.update(deformation=metadata.to_dict(), anchors=anchors,
                              coordinate_units="m", measurement_units="mm", schema_version=2)

    def export_metadata_json(self, metadata, anchors, output_path, *, anchor_units="mm", measured=None):
        factor = 0.001 if anchor_units == "mm" else 1.0
        payload = {**metadata.to_dict(), "coordinate_units": "m", "measurement_units": "mm",
                   "schema_version": 2, "anchors": {k: (np.asarray(v) * factor).tolist() for k, v in anchors.items()}}
        if measured is not None:
            payload["geometry_measurements"] = measured
        output_path = Path(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
