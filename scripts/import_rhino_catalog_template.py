"""Import uneven Rhino eyewear deliveries into the Gold-compatible bundle format.

Connected regions are classified by their canonical bounds, not Rhino object
names. When a source has no separate lens shell, annotated thin lens placeholders
are generated. Every scale is inferred by normalizing the front width to an
eyewear-sized range and is recorded as such in the bundle metadata.
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
CATALOG = {
    "transparent_glasses_with_baked_textures.3dm": ("TG_001", "Transparent Baked-Texture Glasses", "rectangle", "acetate"),
    "sun_glasses.3dm": ("SG_001", "Sunglasses (Model 1)", "geometric", "acetate"),
    "sun_glasses (1).3dm": ("SG_002", "Sunglasses (Model 2)", "geometric", "plastic"),
    "nerd_glasses.3dm": ("NG_001", "Nerd Glasses", "rectangle", "acetate"),
    "hipster_glasses_eyewear_spectacles.3dm": ("HG_001", "Hipster Eyewear Spectacles", "rectangle", "acetate"),
}


def _mesh(rhino_mesh):
    vertices = np.asarray([[v.X, v.Y, v.Z] for v in rhino_mesh.Vertices], dtype=np.float64)
    faces = []
    for face in rhino_mesh.Faces:
        a, b, c, d = list(face)
        faces.append([a, b, c])
        if c != d:
            faces.append([a, c, d])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def _join(meshes):
    meshes = [m for m in meshes if len(m.vertices) and len(m.faces)]
    if not meshes:
        raise ValueError("Source classification left a required named region empty")
    return trimesh.util.concatenate(meshes)


def _lens_placeholder(center_x, center_y, center_z, width, height, thickness):
    """Small elliptical prism, used only when a 3DM has no lens region."""
    n = 48
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[center_x + width * 0.5 * np.cos(angles),
                 center_z + height * 0.5 * np.sin(angles)]
    vertices = np.vstack([
        np.c_[ring[:, 0], np.full(n, center_y - thickness / 2), ring[:, 1]],
        np.c_[ring[:, 0], np.full(n, center_y + thickness / 2), ring[:, 1]],
        [[center_x, center_y - thickness / 2, center_z],
         [center_x, center_y + thickness / 2, center_z]],
    ])
    faces = []
    front_center, back_center = 2 * n, 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[front_center, i, j], [back_center, n + j, n + i],
                      [i, n + i, n + j], [i, n + j, j]])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def convert(source: Path, destination: Path, target_width_mm: float = 145.0):
    model = rhino3dm.File3dm.Read(str(source))
    if model is None or model.Settings.ModelUnitSystem != rhino3dm.UnitSystem.Millimeters:
        raise ValueError("Expected a readable 3DM with millimetre model units")
    mesh_objects = [obj for obj in model.Objects if isinstance(obj.Geometry, rhino3dm.Mesh)]
    raw_meshes = [_mesh(obj.Geometry) for obj in mesh_objects]
    if not raw_meshes:
        raise ValueError("No mesh geometry found in Rhino source")
    raw_vertices = np.vstack([mesh.vertices for mesh in raw_meshes])

    # Align the longest horizontal PCA axis to X, retain world Z as vertical,
    # and derive a perpendicular depth axis. This also recentres translated
    # and slightly rotated deliveries such as the hipster source.
    xy = raw_vertices[:, :2] - raw_vertices[:, :2].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
    axis_x = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis_x[0] < 0:
        axis_x = -axis_x
    axis_y = np.array([-axis_x[1], axis_x[0]])

    raw_components = []
    for object_index, mesh in enumerate(raw_meshes):
        raw_components.extend((object_index, part) for part in mesh.split(only_watertight=False)
                              if len(part.faces) >= 2)

    # Choose depth sign so temple ends point toward positive Y, which is the
    # direction expected by CalibratedRegions when it lengthens the arms.
    rough_x = raw_vertices[:, :2] @ axis_x
    rough_mid_x = (rough_x.min() + rough_x.max()) / 2
    rough_width = float(rough_x.max() - rough_x.min())
    rough_arms = []
    rough_front = []
    for _, part in raw_components:
        p = part.vertices
        cx = float(np.mean(p[:, :2] @ axis_x))
        cy = float(np.mean(p[:, :2] @ axis_y))
        dy = float(np.ptp(p[:, :2] @ axis_y))
        (rough_arms if dy > rough_width * .20 and abs(cx-rough_mid_x) > rough_width*.22 else rough_front).append(cy)
    if rough_arms and rough_front and np.median(rough_arms) < np.median(rough_front):
        axis_y = -axis_y

    def project(vertices):
        return np.column_stack([vertices[:, :2] @ axis_x, vertices[:, :2] @ axis_y, vertices[:, 2]])

    raw_projected = project(raw_vertices)
    raw_bounds = np.stack([raw_projected.min(axis=0), raw_projected.max(axis=0)])
    raw_width = float(raw_bounds[1, 0] - raw_bounds[0, 0])
    scale = target_width_mm / raw_width
    origin = np.array([(raw_bounds[0, 0] + raw_bounds[1, 0]) / 2,
                       0.0, (raw_bounds[0, 2] + raw_bounds[1, 2]) / 2])

    components = []
    for object_index, part in raw_components:
        canonical = project(part.vertices)
        bounds = np.stack([canonical.min(axis=0), canonical.max(axis=0)])
        components.append({"mesh": part, "vertices": canonical, "bounds": bounds,
                           "center": canonical.mean(axis=0), "source_object": object_index})
    if not components:
        raise ValueError("No usable connected mesh regions found")

    # Principal dimensions are robust across the supplied axis-aligned files.
    width = raw_width * scale
    height = float(np.ptp(raw_projected[:, 2]) * scale)
    front_depth = float(np.median([c["center"][1] for c in components]))
    arms = [c for c in components
            if abs(c["center"][0] - origin[0]) > raw_width * 0.22
            and (np.ptp(c["vertices"][:, 1]) > raw_width * 0.20
                 or c["bounds"][1, 1] > front_depth + raw_width * 0.08)]
    if not arms or not any(c["center"][0] > origin[0] for c in arms) or not any(c["center"][0] < origin[0] for c in arms):
        raise ValueError("Could not confidently identify both temple sides")
    arm_ids = {id(c) for c in arms}
    front = [c for c in components if id(c) not in arm_ids]

    # Lens shells are compact, eye-sized connected regions on either side.
    lens_candidates = [c for c in front
        if raw_width * 0.25 < np.ptp(c["vertices"][:, 0]) < raw_width * 0.72
        and np.ptp(c["vertices"][:, 2]) > np.ptp(raw_projected[:, 2]) * 0.35
        and np.ptp(c["vertices"][:, 1]) < raw_width * 0.22
        and abs(c["center"][0] - origin[0]) > raw_width * 0.10]
    left_candidates = [c for c in lens_candidates if c["center"][0] > origin[0]]
    right_candidates = [c for c in lens_candidates if c["center"][0] < origin[0]]
    lens_candidate_ids = {id(c) for c in lens_candidates}
    frame_candidates = [c for c in front if id(c) not in lens_candidate_ids]

    # If extracting lens-shaped components would leave too little frame width,
    # preserve all source front geometry as Frame and add explicitly documented
    # optical surfaces from its dimensions.
    frame_width_raw = (max(c["bounds"][1, 0] for c in frame_candidates) -
                       min(c["bounds"][0, 0] for c in frame_candidates)) if frame_candidates else 0.0
    use_source_lenses = (left_candidates and right_candidates and
                         frame_width_raw >= raw_width * 0.72)
    if use_source_lenses:
        frame_parts = [c["mesh"] for c in frame_candidates]
        left_parts = [c["mesh"] for c in left_candidates]
        right_parts = [c["mesh"] for c in right_candidates]
        lens_y_center = float(np.mean([c["center"][1] for c in lens_candidates]))
        lens_z_center = float(np.mean([c["center"][2] for c in lens_candidates]))
        lens_method = "source connected eye-sized regions; classified by projected bounds and side"
    else:
        frame_parts = [c["mesh"] for c in front]
        lens_parts = []
        lens_y_center = float(np.median([c["center"][1] for c in front])) if front else 0.0
        lens_z_center = origin[2]
        frame_left = (raw_bounds[0, 0] - origin[0]) * scale
        frame_right = (raw_bounds[1, 0] - origin[0]) * scale
        lens_width = min(width * 0.42, height * 1.05)
        bridge = width * 0.16
        center_offset = (lens_width + bridge) / 2
        left_parts = [_lens_placeholder(center_offset, lens_y_center * scale,
                        (lens_z_center-origin[2])*scale, lens_width, height * 0.82, 0.6)]
        right_parts = [_lens_placeholder(-center_offset, lens_y_center * scale,
                         (lens_z_center-origin[2])*scale, lens_width, height * 0.82, 0.6)]
        lens_method = "generated elliptical optical placeholders; source has no separately identifiable lens shells"

    def normalize_mesh(mesh):
        vertices = (project(mesh.vertices) - origin) * scale
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)

    positive_arms = [c["mesh"] for c in arms if c["center"][0] > origin[0]]
    negative_arms = [c["mesh"] for c in arms if c["center"][0] < origin[0]]
    region_by_component = {}
    for c in arms:
        region_by_component[id(c)] = "LeftTemple" if c["center"][0] > origin[0] else "RightTemple"
    if use_source_lenses:
        for c in left_candidates:
            region_by_component[id(c)] = "LeftLens"
        for c in right_candidates:
            region_by_component[id(c)] = "RightLens"
        for c in frame_candidates:
            region_by_component[id(c)] = "Frame"
    else:
        for c in front:
            region_by_component[id(c)] = "Frame"
    named = {
        "Frame": _join([normalize_mesh(m) for m in frame_parts]),
        "LeftLens": _join([normalize_mesh(m) for m in left_parts]) if use_source_lenses else _join(left_parts),
        "RightLens": _join([normalize_mesh(m) for m in right_parts]) if use_source_lenses else _join(right_parts),
        "LeftTemple": _join([normalize_mesh(m) for m in positive_arms]),
        "RightTemple": _join([normalize_mesh(m) for m in negative_arms]),
    }
    for mesh in named.values():
        # Remove sliver triangles whose area is below exporter precision after
        # vertex welding; they can cancel averaged normals at coincident seams.
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-7))
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        if not len(mesh.faces) or not np.isfinite(mesh.vertex_normals).all():
            raise ValueError("A classified region has no valid triangle surface after normal cleanup")

    # Lens centers and bounds supply stable optical/bridge landmarks.
    bounds = {name: mesh.bounds for name, mesh in named.items()}
    left, right = bounds["LeftLens"], bounds["RightLens"]
    bridge_width = float(left[0, 0] - right[1, 0])
    if bridge_width <= 0:
        raise ValueError("Source regions overlap across the bridge after normalizing left/right geometry")
    center_z = float((left[0, 2] + left[1, 2]) / 2)

    def point(v):
        return [round(float(x), 6) for x in v]

    landmarks = {"LM_BridgeCenter": point([0, 0, center_z]),
        "LM_BridgeLeftAttach": point([left[0, 0], 0, center_z]),
        "LM_BridgeRightAttach": point([right[1, 0], 0, center_z]),
        "LM_LeftLensCenter": point([(left[0, 0]+left[1, 0])/2, 0, center_z]),
        "LM_RightLensCenter": point([(right[0, 0]+right[1, 0])/2, 0, center_z])}
    for side, name in (("Left", "LeftLens"), ("Right", "RightLens")):
        b = bounds[name]
        landmarks.update({f"LM_{side}LensTop": point([(b[0,0]+b[1,0])/2,0,b[1,2]]),
                         f"LM_{side}LensBottom": point([(b[0,0]+b[1,0])/2,0,b[0,2]]),
                         f"LM_{side}LensNasal": point([b[0,0] if side=="Left" else b[1,0],0,center_z]),
                         f"LM_{side}LensTemporal": point([b[1,0] if side=="Left" else b[0,0],0,center_z])})
    for side in ("Left", "Right"):
        arr = named[f"{side}Temple"].vertices
        root_y, tip_y = arr[:,1].min(), arr[:,1].max()
        root = arr[arr[:,1] <= root_y + max(0.5, scale*20)].mean(axis=0)
        tip = arr[arr[:,1] >= tip_y - max(0.5, scale*20)].mean(axis=0)
        landmarks[f"LM_{side}TempleRoot"] = point(root)
        landmarks[f"LM_{side}TempleTip"] = point(tip)
        landmarks[f"LM_{side}Hinge"] = point(root)
        landmarks[f"LM_{side}HingeAxis"] = point(root + [0, 0, 1])
        landmarks[f"LM_{side}TemplePivot"] = point(root)

    part_order = list(named)
    slices, offset, all_v, all_f = {}, 0, [], []
    for name in part_order:
        mesh = named[name]
        slices[name] = slice(offset, offset + len(mesh.vertices))
        all_v.append(mesh.vertices)
        all_f.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    rest = np.vstack(all_v)
    faces = np.vstack(all_f).astype(np.int32)
    basis = np.zeros((*rest.shape, 5), dtype=np.float64)
    frame = rest[slices["Frame"]]
    basis[slices["Frame"], 0, 0] = frame[:,0] / max(np.ptp(frame[:,0]), 1e-6)
    for side, name in ((1,"LeftLens"),(-1,"RightLens")):
        sl=slices[name]; lens=rest[sl]
        cx=(lens[:,0].min()+lens[:,0].max())/2; cz=(lens[:,2].min()+lens[:,2].max())/2
        basis[sl,0,1]=(lens[:,0]-cx)/max(np.ptp(lens[:,0]),1e-6)
        basis[sl,2,2]=(lens[:,2]-cz)/max(np.ptp(lens[:,2]),1e-6)
        basis[sl,0,3]=side*0.5

    measurements = {"frame_width": float(np.ptp(named["Frame"].vertices[:,0])),
        "lens_width": float(np.ptp(named["LeftLens"].vertices[:,0])),
        "lens_height": float(np.ptp(named["LeftLens"].vertices[:,2])),
        "bridge_width": bridge_width,
        "temple_length": float(np.linalg.norm(np.asarray(landmarks["LM_LeftTempleTip"])-landmarks["LM_LeftTempleRoot"]))}
    if measurements["frame_width"] < 2*measurements["lens_width"] + bridge_width:
        # Preserve the measured front width when a source's material groups do
        # not include full outer rims; increase metadata frame bounds as a note.
        measurements["frame_width"] = width
    profile = CATALOG[source.name]
    template_id, display_name, shape, material = profile
    scene = trimesh.Scene(base_frame="world")
    for name, mesh in named.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.metadata.update(coordinate_units="mm", template_id=template_id,
                          source_units="millimeters", source_scale_factor=scale)
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ("geometry", "deformation", "metadata", "source"):
        (destination/folder).mkdir(exist_ok=True)
    scene.export(destination/"geometry/template.glb")
    shutil.copy2(source, destination/"source"/source.name)
    np.savez_compressed(destination/"deformation/basis.npz", V0=rest, B=basis, faces=faces,
                        part_order=np.asarray(part_order))
    (destination/"deformation/_part_order.json").write_text(json.dumps(part_order,indent=2)+"\n",encoding="utf-8")

    param_specs = [("frame_width",max(100,measurements["frame_width"]-20),min(190,measurements["frame_width"]+20)),
        ("lens_width",max(30,measurements["lens_width"]-12),min(80,measurements["lens_width"]+12)),
        ("lens_height",max(20,measurements["lens_height"]-12),min(65,measurements["lens_height"]+12)),
        ("bridge_width",max(8,bridge_width-8),min(35,bridge_width+8)),
        ("temple_length",max(60,measurements["temple_length"]-25),min(200,measurements["temple_length"]+25))]
    params=[{"name":n,"units":"mm","default":round(measurements[n],6),"min":round(lo,6),"max":round(hi,6),"affected_parts":part_order}
            for n,lo,hi in param_specs]
    (destination/"deformation/basis_metadata.json").write_text(json.dumps({"parameters":params,
        "formula":"V(p) = V0 + sum_i B[:,:,i] * (p_i - default_i); optical dimensions use measured Jacobian calibration.",
        "units":"millimeters","part_order":part_order,"version":"1.0"},indent=2)+"\n",encoding="utf-8")
    raw_frame_width = width
    template={"template_id":template_id,"display_name":display_name,"template_version":"1.0.0",
        "shape":shape,"material":material,"frame_family":shape,"rim_type":"full_rim",
        "bridge_type":"saddle","parts":part_order,"tags":[template_id.lower(),shape,material],
        "description":f"Imported and annotated from {source.name}; see source_objects.json and scale_config.json for classification assumptions.",
        "dimensions":{**{k:round(v,6) for k,v in measurements.items()},"rim_thickness":1.2},
        "source_file":f"source/{source.name}","source_model_units":"mm","source_geometry_scale":scale,
        "landmark_method":"Lens bounds and temple end vertex clusters in PCA-aligned canonical eyewear coordinates.",
        "lens_geometry_method":lens_method,"geometry_object_names":part_order}
    (destination/"metadata/template.json").write_text(json.dumps(template,indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/landmarks.json").write_text(json.dumps({"landmarks":landmarks,
        "coordinate_system":{"x":"left_right; positive is Left","y":"front_back","z":"up_down","units":"mm"},
        "derivation":"Optical landmarks use named lens bounds. Hinge pivots use temple roots because source hinge landmarks are absent."},indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/measurements.json").write_text(json.dumps(measurements,indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/parameter_schema.json").write_text(json.dumps({"parameters":params},indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/constraints.json").write_text(json.dumps({"coupled":[{"formula":"frame_width >= 2*lens_width + bridge_width + 2*rim_thickness","description":"Reserves frame and bridge clearance on both sides."}]},indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/topology.json").write_text(json.dumps({"coordinate_system":{"x":"left_right","y":"front_back","z":"up_down","units":"mm"},
        "parts":{n:{"vertices":len(m.vertices),"triangles":len(m.faces)} for n,m in named.items()}},indent=2)+"\n",encoding="utf-8")
    ranges={n:{"start":int(s.start),"stop":int(s.stop)} for n,s in slices.items()}
    (destination/"metadata/masks.json").write_text(json.dumps({"unified_vertex_order":part_order,"vertex_count":len(rest),"part_masks":ranges},indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/region_masks.json").write_text(json.dumps({"regions":{n:{"parts":[n],"vertex_range":ranges[n]} for n in part_order},
        "front_frame":{"parts":["Frame"]},"optical":{"parts":["LeftLens","RightLens"]},"temples":{"parts":["LeftTemple","RightTemple"]}},indent=2)+"\n",encoding="utf-8")
    (destination/"metadata/scale_config.json").write_text(json.dumps({"source_units":"millimeters","runtime_units":"millimeters",
        "source_geometry_scale_factor":scale,"raw_frame_width_model_units":round(raw_frame_width/scale,6),
        "normalized_frame_width_mm":round(width,6),"scale_rationale":f"Inferred uniform scale {scale:.9g} normalizes the horizontal model extent to {target_width_mm:g} mm; verify against product dimensions."},indent=2)+"\n",encoding="utf-8")
    regions_by_object = {i:set() for i in range(len(mesh_objects))}
    for c in components:
        region = region_by_component.get(id(c))
        if region:
            regions_by_object[c["source_object"]].add(region)
    (destination/"metadata/source_objects.json").write_text(json.dumps({"source_file":f"source/{source.name}",
        "objects":[{"source_object":f"Object_{i}","source_name":o.Attributes.Name or f"Object_{i}",
          "geometry_type":"Mesh","renamed_regions":sorted(regions_by_object[i])}
          for i,o in enumerate(mesh_objects)],
        "classification":"Connected components classified from projected bounds; see lens_geometry_method and scale_config.",
        "notes":"Source has no standard LM_* point/empty objects. Landmark JSON records inferred points; generated optical surfaces are explicitly identified when used."},indent=2)+"\n",encoding="utf-8")
    readme=(f"# {display_name} ({template_id})\n\nImported from `{source.name}` and registered as a basis deformation template beside GT_001. "
        f"The source declares millimetres; scale {scale:.9g} is inferred by fitting its horizontal extent to {target_width_mm:g} mm. "
        "Check this against a known product measurement before claiming physical calibration. Source-region names, landmark derivations, "
        "lens handling, and mesh counts are recorded in the metadata JSON files.\n")
    (destination/"README.md").write_text(readme,encoding="utf-8")
    return {"template":template_id,"measurements_mm":measurements,"lens_geometry_method":lens_method,
            "parts":{n:{"vertices":len(m.vertices),"triangles":len(m.faces)} for n,m in named.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--target-width-mm",type=float,default=145.0)
    args=parser.parse_args()
    ident=CATALOG.get(args.source.name)
    if ident is None: raise SystemExit(f"No catalog annotation profile for {args.source.name}")
    destination=args.output or REPO/"assets"/"templates"/ident[0]
    print(json.dumps(convert(args.source,destination,args.target_width_mm),indent=2))

if __name__=="__main__": main()
