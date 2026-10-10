"""Normalize an eyewear Rhino mesh into the runtime deformation bundle.

Ray-Ban's supplied 3DM is an authoring mesh, not a runtime template. This
converter names its disconnected lens, front, temple and pad regions, installs
the Gold-compatible metadata contract, and writes a five-parameter basis.
Requires rhino3dm (development dependency); runtime does not read 3DM files.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rhino3dm
import trimesh


REPO = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "RB_001"
PARAMETERS = [
    ("frame_width", 110.0, 170.0),
    ("lens_width", 40.0, 70.0),
    ("lens_height", 25.0, 55.0),
    ("bridge_width", 12.0, 28.0),
    ("temple_length", 120.0, 180.0),
]


def _rhino_mesh(mesh):
    vertices = np.asarray([[v.X, v.Y, v.Z] for v in mesh.Vertices], dtype=np.float64)
    faces = []
    for a, b, c, d in mesh.Faces:
        faces.append([a, b, c])
        if c != d:
            faces.append([a, c, d])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _joined(parts):
    if not parts:
        raise ValueError("Rhino classification produced an empty required region")
    return trimesh.util.concatenate(parts)


def convert(source: Path, destination: Path, scale: float):
    model = rhino3dm.File3dm.Read(str(source))
    if model is None or model.Settings.ModelUnitSystem != rhino3dm.UnitSystem.Millimeters:
        raise ValueError("Expected a readable Rhino 3DM whose model units are millimetres")
    objects = [_rhino_mesh(obj.Geometry) for obj in model.Objects
               if isinstance(obj.Geometry, rhino3dm.Mesh)]
    def side_parts(parts, side):
        return [p for p in parts if (p.centroid[0] >= 0) == (side > 0)]

    hinges = {-1: [], 1: []}
    if len(objects) == 2:
        # This delivery has an unnamed frame/temple object and a second object
        # containing lens shells and nose-pad parts. Classify components by
        # physical extent and side instead of trusting generic Rhino names.
        frame_components, optical_components = [o.split(only_watertight=False) for o in objects]
        # Rhino exported the arms as one large component plus many unwelded
        # four-vertex patches. Include every lateral patch extending behind
        # the front; taking only the two largest connected shells leaves the
        # arm tips in the fixed region and breaks temple-length deformation.
        temples = [part for part in frame_components
                   if part.bounds[1, 1] > -500.0 and abs(part.centroid[0]) > 600.0]
        if not temples or not side_parts(temples, 1) or not side_parts(temples, -1):
            raise ValueError("Could not isolate both temple regions from the source mesh")
        temple_ids = {id(part) for part in temples}
        front = [part for part in frame_components if id(part) not in temple_ids]
        lens_components = [part for part in optical_components if np.ptp(part.vertices[:, 0]) > 400.0]
        lens_ids = {id(part) for part in lens_components}
        pad_components = [part for part in optical_components if id(part) not in lens_ids]
        if len(lens_components) != 4:
            raise ValueError(f"Expected four lens shell patches (two per lens), found {len(lens_components)}")
        if len(side_parts(lens_components, 1)) != 2 or len(side_parts(lens_components, -1)) != 2:
            raise ValueError("Lens shell patches are not paired by side")
        if not side_parts(pad_components, 1) or not side_parts(pad_components, -1):
            raise ValueError("Could not identify nose-pad components on both sides")
        positive_lens = _joined(side_parts(lens_components, 1))
        negative_lens = _joined(side_parts(lens_components, -1))
    elif len(objects) == 3:
        # Keep compatibility with the earlier 3-mesh delivery.
        lens_components = objects[0].split(only_watertight=False)
        frame_components = objects[1].split(only_watertight=False)
        pad_components = objects[2].split(only_watertight=False)
        if len(lens_components) != 2:
            raise ValueError(f"Expected two disconnected lens shells, found {len(lens_components)}")
        temples = [part for part in frame_components if np.ptp(part.vertices[:, 1]) > 100.0]
        temple_ids = {id(part) for part in temples}
        front_and_hardware = [part for part in frame_components if id(part) not in temple_ids]
        front = []
        for part in front_and_hardware:
            b, center = part.bounds, part.centroid
            side = 1 if center[0] >= 0 else -1
            is_hinge = abs(center[0]) > 750 and b[1, 1] < 60 and 145 < center[2] < 205 and len(part.vertices) <= 20
            (hinges[side] if is_hinge else front).append(part)
        if len(temples) != 6:
            raise ValueError("Unexpected disconnected temple regions in source 3DM")
        positive_lens = max(lens_components, key=lambda p: p.centroid[0])
        negative_lens = min(lens_components, key=lambda p: p.centroid[0])
    else:
        raise ValueError(f"Expected the 2-mesh delivery (or legacy 3-mesh file), found {len(objects)} meshes")

    # X/Z are the eyewear front plane; Y is temple depth. The new source
    # declares mm but spans about 1,914 units across the front. The 0.08 factor
    # yields a plausible 153 mm front and remains explicitly inferred.
    raw_front = _joined(front)
    frame_center_x = float((raw_front.bounds[0, 0] + raw_front.bounds[1, 0]) / 2)
    lens_center_z = float((positive_lens.bounds[0, 2] + positive_lens.bounds[1, 2] + negative_lens.bounds[0, 2] + negative_lens.bounds[1, 2]) / 4)
    lens_center_y = float((positive_lens.bounds[0, 1] + positive_lens.bounds[1, 1] + negative_lens.bounds[0, 1] + negative_lens.bounds[1, 1]) / 4)
    origin = np.array([frame_center_x, lens_center_y, lens_center_z])
    transform = lambda mesh: trimesh.Trimesh(
        vertices=(mesh.vertices - origin) * scale, faces=mesh.faces.copy(), process=False)

    named = {
        "Frame": transform(_joined(front)),
        "LeftLens": transform(positive_lens),
        "RightLens": transform(negative_lens),
        "LeftTemple": transform(_joined(side_parts(temples, 1))),
        "RightTemple": transform(_joined(side_parts(temples, -1))),
        "NosePadLeft": transform(_joined(side_parts(pad_components, 1))),
        "NosePadRight": transform(_joined(side_parts(pad_components, -1))),
    }
    if hinges[1] and hinges[-1]:
        named["LeftHinge"] = transform(_joined(hinges[1]))
        named["RightHinge"] = transform(_joined(hinges[-1]))
    if any(len(mesh.faces) == 0 for mesh in named.values()):
        raise ValueError("A named Rhino region contains no triangles")

    scene = trimesh.Scene(base_frame="world")
    for name, mesh in named.items():
        mesh.fix_normals()
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.metadata.update(coordinate_units="mm", template_id=TEMPLATE_ID,
                          source_units="millimeters", source_scale_factor=scale,
                          source_origin_mm=origin.tolist())

    part_order = list(named)
    rest = np.vstack([named[name].vertices for name in part_order])
    faces, slices, offset = [], {}, 0
    for name in part_order:
        mesh = named[name]
        faces.append(mesh.faces + offset)
        slices[name] = slice(offset, offset + len(mesh.vertices))
        offset += len(mesh.vertices)
    faces = np.vstack(faces).astype(np.int32)

    # Basis columns are derivatives in mm/mm. The calibrated-region engine
    # measures the resulting 4x4 Jacobian and applies its inverse at runtime.
    basis = np.zeros((*rest.shape, 5), dtype=np.float64)
    frame = rest[slices["Frame"]]
    basis[slices["Frame"], 0, 0] = frame[:, 0] / np.ptp(frame[:, 0])
    for side, lens_name in ((1, "LeftLens"), (-1, "RightLens")):
        sl = slices[lens_name]
        lens = rest[sl]
        center_x = (lens[:, 0].min() + lens[:, 0].max()) / 2
        center_z = (lens[:, 2].min() + lens[:, 2].max()) / 2
        basis[sl, 0, 1] = (lens[:, 0] - center_x) / np.ptp(lens[:, 0])
        basis[sl, 2, 2] = (lens[:, 2] - center_z) / np.ptp(lens[:, 2])
        basis[sl, 0, 3] = side * 0.5

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "geometry").mkdir(exist_ok=True)
    (destination / "deformation").mkdir(exist_ok=True)
    (destination / "metadata").mkdir(exist_ok=True)
    (destination / "source").mkdir(exist_ok=True)
    scene.export(destination / "geometry/template.glb")
    source_name = "ray_ban_glasses.3dm"
    shutil.copy2(source, destination / "source" / source_name)
    np.savez_compressed(destination / "deformation/basis.npz", V0=rest, B=basis,
                        faces=faces, part_order=np.asarray(part_order))
    (destination / "deformation/_part_order.json").write_text(
        json.dumps(part_order, indent=2) + "\n", encoding="utf-8")

    def bounds(name):
        return named[name].bounds

    def point(x, y, z):
        return [round(float(x), 6), round(float(y), 6), round(float(z), 6)]

    positive, negative = bounds("LeftLens"), bounds("RightLens")
    landmarks = {
        "LM_BridgeCenter": point(0, 0, 0),
        "LM_BridgeLeftAttach": point(positive[0, 0], 0, 0),
        "LM_BridgeRightAttach": point(negative[1, 0], 0, 0),
        "LM_LeftLensCenter": point((positive[0, 0] + positive[1, 0]) / 2, 0,
                                    (positive[0, 2] + positive[1, 2]) / 2),
        "LM_RightLensCenter": point((negative[0, 0] + negative[1, 0]) / 2, 0,
                                     (negative[0, 2] + negative[1, 2]) / 2),
    }
    for side, name in (("Left", "LeftLens"), ("Right", "RightLens")):
        b = bounds(name)
        nasal_x, temporal_x = ((b[0, 0], b[1, 0]) if side == "Left" else (b[1, 0], b[0, 0]))
        landmarks.update({
            f"LM_{side}LensTop": point((b[0, 0] + b[1, 0]) / 2, 0, b[1, 2]),
            f"LM_{side}LensBottom": point((b[0, 0] + b[1, 0]) / 2, 0, b[0, 2]),
            f"LM_{side}LensNasal": point(nasal_x, 0, (b[0, 2] + b[1, 2]) / 2),
            f"LM_{side}LensTemporal": point(temporal_x, 0, (b[0, 2] + b[1, 2]) / 2),
        })
    for side in ("Left", "Right"):
        temple = named[f"{side}Temple"].vertices
        root_y, tip_y = temple[:, 1].min(), temple[:, 1].max()
        root = temple[temple[:, 1] <= root_y + 1.5].mean(axis=0)
        tip = temple[temple[:, 1] >= tip_y - 1.5].mean(axis=0)
        hinge_name = f"{side}Hinge"
        hinge = named[hinge_name].vertices.mean(axis=0) if hinge_name in named else root
        landmarks[f"LM_{side}Hinge"] = point(*hinge)
        landmarks[f"LM_{side}HingeAxis"] = point(hinge[0], hinge[1], hinge[2] + 1.0)
        landmarks[f"LM_{side}TempleRoot"] = point(*root)
        landmarks[f"LM_{side}TempleTip"] = point(*tip)
        landmarks[f"LM_{side}TemplePivot"] = point(*hinge)
    for side in ("Left", "Right"):
        landmarks[f"LM_NosePad{side}Center"] = point(*named[f"NosePad{side}"].vertices.mean(axis=0))

    measurements = {
        "frame_width": float(np.ptp(named["Frame"].vertices[:, 0])),
        "lens_width": float(np.ptp(named["LeftLens"].vertices[:, 0])),
        "lens_height": float(np.ptp(named["LeftLens"].vertices[:, 2])),
        "bridge_width": float(named["LeftLens"].vertices[:, 0].min() - named["RightLens"].vertices[:, 0].max()),
        "temple_length": float(np.linalg.norm(np.asarray(landmarks["LM_LeftTempleTip"]) - landmarks["LM_LeftTempleRoot"])),
    }
    pad_left, pad_right = named["NosePadLeft"].vertices.mean(axis=0), named["NosePadRight"].vertices.mean(axis=0)
    supplemental_dimensions = {
        "nose_pad_distance": float(abs(pad_left[0] - pad_right[0])),
        "nose_pad_height": float(np.ptp(named["NosePadLeft"].vertices[:, 2])),
        "temple_curve_angle": float(np.degrees(np.arctan2(
            abs(landmarks["LM_LeftTempleTip"][2] - landmarks["LM_LeftTempleRoot"][2]),
            abs(landmarks["LM_LeftTempleTip"][1] - landmarks["LM_LeftTempleRoot"][1])))),
    }
    defaults = {name: round(measurements[name], 6) for name, _, _ in PARAMETERS}
    parameter_meta = []
    part_names = list(named)
    for name, lower, upper in PARAMETERS:
        parameter_meta.append({"name": name, "units": "mm", "default": defaults[name],
                               "min": lower, "max": upper, "affected_parts": part_names})
    bundle = {
        "parameters": parameter_meta,
        "formula": "V(p) = V0 + sum_i B[:,:,i] * (p_i - default_i); dimensions are calibrated from the measured basis Jacobian.",
        "units": "millimeters", "part_order": part_order,
        "landmark_source": "Calculated from the named source mesh regions; see metadata/landmarks.json derivation fields.",
        "version": "1.0",
    }
    common = {"shape": "rectangle", "material": "acetate", "frame_family": "wayfarer",
              "rim_type": "full_rim", "bridge_type": "pad", "parts": part_names,
              "tags": ["ray-ban", "wayfarer", "rectangle", "acetate", "full-rim"]}
    template = {"template_id": TEMPLATE_ID, "display_name": "Ray-Ban Wayfarer (RB_001)",
                "template_version": "1.0.0", **common,
                "description": "Ray-Ban frame normalized from the supplied Rhino mesh, with named runtime regions and an mm deformation basis.",
                "dimensions": {**{name: round(value, 6) for name, value in measurements.items()},
                               **{name: round(value, 6) for name, value in supplemental_dimensions.items()}},
                "source_file": f"source/{source_name}", "source_model_units": "mm",
                "source_geometry_scale": scale, "source_frame_width_mm": round(measurements["frame_width"], 6),
                "landmark_method": "Lens bounding boxes define optical landmarks; root/tip coordinates are averaged from the frontmost and rearmost temple mesh vertices; hinge centers are mesh centroids.",
                "geometry_object_names": part_names}
    (destination / "deformation/basis_metadata.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/template.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/measurements.json").write_text(json.dumps(
        {**measurements, **supplemental_dimensions}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/parameter_schema.json").write_text(json.dumps({"parameters": parameter_meta}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/constraints.json").write_text(json.dumps({"coupled": [
        {"formula": "frame_width >= 2*lens_width + bridge_width + 2*rim_thickness",
         "description": "Reserves positive outer frame clearance on both sides."}]}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/topology.json").write_text(json.dumps({
        "coordinate_system": {"x": "left_right", "y": "front_back", "z": "up_down", "units": "mm"},
        "parts": {name: {"vertices": len(mesh.vertices), "triangles": len(mesh.faces)}
                  for name, mesh in named.items()}}, indent=2) + "\n", encoding="utf-8")
    part_ranges = {name: {"start": int(sl.start), "stop": int(sl.stop)} for name, sl in slices.items()}
    masks = {"unified_vertex_order": part_order, "vertex_count": len(rest), "part_masks": part_ranges}
    (destination / "metadata/masks.json").write_text(json.dumps(masks, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/region_masks.json").write_text(json.dumps({
        "regions": {name: {"parts": [name], "vertex_range": part_ranges[name]} for name in part_order},
        "front_frame": {"parts": [name for name in ("Frame", "LeftHinge", "RightHinge") if name in named]},
        "optical": {"parts": ["LeftLens", "RightLens"]},
        "temples": {"parts": ["LeftTemple", "RightTemple"]},
        "nose_pads": {"parts": ["NosePadLeft", "NosePadRight"]}}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/scale_config.json").write_text(json.dumps({
        "source_units": "millimeters", "source_model_unit_system": "Millimeters",
        "source_geometry_scale_factor": scale, "runtime_units": "millimeters",
        "raw_frame_width_model_units": round(float(np.ptp(raw_front.vertices[:, 0])), 6),
        "normalized_frame_width_mm": round(measurements["frame_width"], 6),
        "scale_rationale": f"The supplied file declares mm but has a {np.ptp(raw_front.vertices[:, 0]):.1f}-unit front frame; factor {scale:g} yields {measurements['frame_width']:.2f} mm, a plausible eyewear width. This is inferred and must be checked against a product specification."}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/source_objects.json").write_text(json.dumps({
        "source_file": f"source/{source_name}",
        "objects": ([
            {"source_object": "Object_0", "source_name": "defaultMaterial", "renamed_regions": ["Frame", "LeftTemple", "RightTemple"]},
            {"source_object": "Object_1", "source_name": "defaultMaterial", "renamed_regions": ["LeftLens", "RightLens", "NosePadLeft", "NosePadRight"]}]
            if len(objects) == 2 else [
            {"source_object": "Object_0", "renamed_regions": ["LeftLens", "RightLens"]},
            {"source_object": "Object_1", "renamed_regions": ["Frame", "LeftTemple", "RightTemple", "LeftHinge", "RightHinge"]},
            {"source_object": "Object_2", "renamed_regions": ["NosePadLeft", "NosePadRight"]}]),
        "note": "The source has no named point/empty objects or useful object properties; landmarks and region names are generated from mesh components."}, indent=2) + "\n", encoding="utf-8")
    (destination / "metadata/landmarks.json").write_text(json.dumps({
        "landmarks": landmarks,
        "derivation": "Landmarks were generated from mesh-region bounds/vertex clusters because the source 3DM contains only unnamed mesh objects and no named point objects."}, indent=2) + "\n", encoding="utf-8")
    return measurements, named


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path.home() / "Downloads/ray_ban_glasses.3dm")
    parser.add_argument("--output", type=Path, default=REPO / "assets/templates/RB_001")
    parser.add_argument("--scale", type=float, default=0.08,
                        help="Physical mm per Rhino coordinate; inferred default yields a 153 mm frame")
    args = parser.parse_args()
    measurements, meshes = convert(args.source, args.output, args.scale)
    print(json.dumps({"template": TEMPLATE_ID, "measurements_mm": measurements,
                      "parts": {name: {"vertices": len(mesh.vertices), "triangles": len(mesh.faces)}
                                for name, mesh in meshes.items()}}, indent=2))


if __name__ == "__main__":
    main()
