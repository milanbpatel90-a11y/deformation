"""Shared tooling to turn a raw eyewear GLB into an engine template.

The engine contract (see rectangle_plastic.glb / geometric_metal.glb):

* 8 physically separate, distinctly named meshes
  Frame, Bridge, LeftRim, RightRim, LeftLens, RightLens, LeftTemple, RightTemple
* millimetres, Y up, front of the frame at +Z (temples run toward -Z), Left = -X, centred on X = 0
* empty marker nodes Bridge_Center, Left_Hinge, Right_Hinge, Left_Temple_End, Right_Temple_End
* a rim whose projected footprint has a closed opening that contains the lens
  (``LensDeformer._extract_rim_boundary`` fits the requested lens into that opening)

Raw assets never satisfy this, so each template has a small ``split_<name>.py`` that decides
which connected components are lens / front / temple / hardware (by geometry, never by node
name) and calls the helpers here to cut the front, validate the rim openings, write the GLB and
write ``templates/<name>.json`` + ``templates/descriptors/<name>.json``.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.deformer.lens_deformer import LensDeformer  # noqa: E402

PART_NAMES = ("Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens", "LeftTemple", "RightTemple")


# --------------------------------------------------------------------------- loading / basic ops
def rotation_y(degrees: float) -> np.ndarray:
    """4x4 rotation about +Y (right-handed): x' = x cos + z sin, z' = -x sin + z cos."""
    a = np.radians(degrees)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = np.cos(a), np.sin(a), -np.sin(a), np.cos(a)
    return m


def load_components(
    source: Path,
    *,
    scale: float = 1.0,
    transform: np.ndarray | None = None,
    drop: Callable[[trimesh.Trimesh], bool] | None = None,
    min_faces: int = 1,
    weld_seams: bool = False,
) -> list[trimesh.Trimesh]:
    """Bake node transforms, apply ``transform`` then ``scale`` (to mm), return connected components.

    Each component keeps ``metadata['src']`` (source node name) so splitters can use it.
    """
    scene = trimesh.load(str(source), force="scene")
    out: list[trimesh.Trimesh] = []
    for node in scene.graph.nodes_geometry:
        matrix, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(matrix)
        if transform is not None:
            mesh.apply_transform(transform)
        mesh.vertices = np.asarray(mesh.vertices) * scale
        if weld_seams:   # ignore UV/normal splits so texture seams weld by position (round_metal needs this)
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
        else:
            mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        for comp in mesh.split(only_watertight=False):
            if len(comp.faces) < min_faces or (drop and drop(comp)):
                continue
            comp.metadata["src"] = node
            comp.metadata["geom"] = geom_name
            out.append(comp)
    return out


def cat(meshes: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = list(meshes)
    if not meshes:
        raise ValueError("cannot concatenate zero meshes")
    return meshes[0].copy() if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def sub(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    if not np.asarray(mask).any():
        raise ValueError("empty slice")
    part = mesh.submesh([np.where(mask)[0]], append=True, repair=False)
    part.remove_unreferenced_vertices()
    return part


def side_of(mesh: trimesh.Trimesh) -> str:
    return "Left" if mesh.bounds.mean(axis=0)[0] < 0 else "Right"


def mirror_x(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Mirror across X=0 (flips winding so normals stay outward)."""
    m = mesh.copy()
    v = np.asarray(m.vertices).copy()
    v[:, 0] *= -1
    m.vertices = v
    m.faces = m.faces[:, ::-1]
    return m


# --------------------------------------------------------------------------- front cutting
def cut_front(front: trimesh.Trimesh, bridge_half: float, end_x: float) -> dict[str, trimesh.Trimesh]:
    cx = front.triangles_center[:, 0]
    return {
        "Bridge": sub(front, np.abs(cx) < bridge_half),
        "LeftRim": sub(front, (cx <= -bridge_half) & (cx > -end_x)),
        "RightRim": sub(front, (cx >= bridge_half) & (cx < end_x)),
        "Frame": sub(front, np.abs(cx) >= end_x),
    }


def aperture_status(rim: trimesh.Trimesh, lens: trimesh.Trimesh, side: str) -> tuple[bool, str, float]:
    """Run the engine's own aperture extraction. Returns (ok, message, aperture_area)."""
    helper = LensDeformer()
    ctx = SimpleNamespace(mesh=lambda n: {"R": rim, "L": lens}[n])
    sel = SimpleNamespace(rim_part="R", lens_part="L", side=side.lower())
    try:
        from shapely.geometry import Polygon
        ring = helper._extract_rim_boundary(ctx, sel)
        return True, "ok", float(Polygon(ring).area)
    except Exception as exc:  # noqa: BLE001 - we report the reason
        return False, str(exc)[:160], 0.0


def search_cut(
    front: trimesh.Trimesh,
    lens: dict[str, trimesh.Trimesh],
    *,
    bridge_start: float | None = None,
    end_start: float | None = None,
    step: float = 0.5,
    max_rim_ratio: float = 0.55,
) -> tuple[float, float, dict[str, trimesh.Trimesh]]:
    """Find (bridge_half, end_x) so both rims expose a lens-sized aperture and the bridge/end
    pieces stay outside it. Starts just inside the lens' nasal edge / just outside its temporal
    edge and walks outward/inward until the engine accepts the aperture."""
    inner = min(abs(lens["Left"].bounds[1][0]), abs(lens["Right"].bounds[0][0]))   # nasal edges
    outer = max(abs(lens["Left"].bounds[0][0]), abs(lens["Right"].bounds[1][0]))   # temporal edges
    half_front = float(max(abs(front.bounds[0][0]), abs(front.bounds[1][0])))
    b0 = bridge_start if bridge_start is not None else inner - 1.5
    e0 = end_start if end_start is not None else outer + 1.3
    tried = []
    for db in np.arange(0.0, max(b0 - 2.0, 1.0), step):
        b = b0 - db
        if b < 2.0:
            break
        for de in np.arange(0.0, max(half_front - e0, step) + 1e-9, step):
            e = e0 + de
            if e > half_front - 0.2:
                break
            try:
                parts = cut_front(front, b, e)
            except ValueError:
                continue
            ok_all, why = True, ""
            for s in ("Left", "Right"):
                ok, msg, _ = aperture_status(parts[f"{s}Rim"], lens[s], s)
                if not ok:
                    ok_all, why = False, msg
                    break
            span = np.ptp(np.vstack([parts["LeftRim"].vertices, parts["RightRim"].vertices])[:, 0])
            rim_w = max(np.ptp(parts[f"{s}Rim"].vertices[:, 0]) for s in ("Left", "Right"))
            if ok_all and rim_w / max(span, 1e-6) <= max_rim_ratio + 0.07:
                return float(b), float(e), parts
            tried.append((round(b, 2), round(e, 2), why))
    raise ValueError(f"no bridge/end cut gives a valid rim aperture; last attempts: {tried[-3:]}")


def synthesize_lenses(front: trimesh.Trimesh, thickness: float = 2.0, clearance: float = 0.5) -> dict[str, trimesh.Trimesh]:
    """For assets that ship WITHOUT lenses: build one flat lens per rim opening.

    The openings are the interior holes of the front's projected footprint (the same footprint the
    engine uses). Each lens is that opening inset by ``clearance`` mm and extruded ``thickness`` mm,
    centred on the Z of the rim material around the opening."""
    from shapely.geometry import Polygon

    foot = LensDeformer._projected_footprint(front, "front")
    holes: list[Polygon] = []
    for poly in LensDeformer._polygon_components(foot):
        holes += [Polygon(r) for r in poly.interiors]
    holes = sorted((h for h in holes if h.area > 100.0), key=lambda h: -h.area)[:2]
    if len(holes) != 2:
        raise ValueError(f"expected 2 lens openings in the front, found {len(holes)}")
    out: dict[str, trimesh.Trimesh] = {}
    verts = np.asarray(front.vertices)
    for hole in holes:
        poly = hole.buffer(-clearance)
        if poly.is_empty or poly.geom_type != "Polygon":
            raise ValueError("lens opening collapsed after inset")
        ring = np.asarray(hole.exterior.coords)
        d = np.min(np.linalg.norm(verts[:, None, :2] - ring[None, ::4, :], axis=2), axis=1)
        z = float(np.median(verts[d < 2.5, 2])) if (d < 2.5).any() else float(verts[:, 2].mean())
        lens = trimesh.creation.extrude_polygon(poly, thickness)
        lens.apply_translation([0, 0, z - thickness / 2.0])
        out["Left" if hole.centroid.x < 0 else "Right"] = lens
    if set(out) != {"Left", "Right"}:
        raise ValueError("both lens openings are on the same side of X=0 (centre the asset first)")
    return out


# --------------------------------------------------------------------------- scene assembly
def default_markers(parts: dict[str, trimesh.Trimesh]) -> dict[str, np.ndarray]:
    """Hinge = front end of each temple (its +Z end); tip = farthest point behind it."""
    markers: dict[str, np.ndarray] = {"Bridge_Center": parts["Bridge"].bounds.mean(axis=0)}
    for s in ("Left", "Right"):
        v = np.asarray(parts[f"{s}Temple"].vertices)
        near = v[v[:, 2] >= v[:, 2].max() - 3.0]
        markers[f"{s}_Hinge"] = near.mean(axis=0)
        markers[f"{s}_Temple_End"] = v[np.argmin(v[:, 2])]
    return markers


def build_scene(parts: dict[str, trimesh.Trimesh], markers: dict[str, np.ndarray] | None = None) -> trimesh.Scene:
    missing = [p for p in PART_NAMES if p not in parts]
    if missing:
        raise ValueError(f"missing parts: {missing}")
    if len({id(parts[p]) for p in PART_NAMES}) != len(PART_NAMES):
        raise ValueError("parts must be distinct mesh objects")
    markers = markers or default_markers(parts)
    scene = trimesh.Scene()
    for name in PART_NAMES:
        scene.add_geometry(parts[name], node_name=name, geom_name=name)
    for name, point in markers.items():
        m = np.eye(4)
        m[:3, 3] = point
        scene.graph.update(frame_to=name, frame_from=scene.graph.base_frame, matrix=m)
    return scene


def centre_x(parts: dict[str, trimesh.Trimesh]) -> None:
    """Centre the whole assembly on X=0 using the front bounds (parts are modified in place)."""
    xs = np.vstack([parts[p].bounds for p in ("Frame", "LeftRim", "RightRim", "Bridge")])
    dx = -0.5 * (xs[:, 0].min() + xs[:, 0].max())
    for m in parts.values():
        m.apply_translation([dx, 0, 0])


def fit_real_size(parts: dict[str, trimesh.Trimesh], front_width: float | None, temple_length: float | None) -> None:
    """Bring a raw asset to eyewear scale. The deformation engine never resizes the overall front
    width, so a template authored at 170 mm would stay 170 mm whatever was requested."""
    if front_width:
        front = np.vstack([parts[p].vertices for p in ("Frame", "LeftRim", "RightRim", "Bridge")])
        s = front_width / float(np.ptp(front[:, 0]))
        hinge_z = float(np.mean([parts[f"{x}Temple"].vertices[:, 2].max() for x in ("Left", "Right")]))
        pivot = np.array([0.0, float(front[:, 1].mean()), hinge_z])
        for m in parts.values():
            m.vertices = pivot + (np.asarray(m.vertices) - pivot) * s
    if temple_length:
        cur = max(np.ptp(parts[f"{x}Temple"].vertices[:, 2]) for x in ("Left", "Right"))
        k = temple_length / float(cur)
        hz = float(np.mean([parts[f"{x}Temple"].vertices[:, 2].max() for x in ("Left", "Right")]))
        for x in ("Left", "Right"):
            v = np.asarray(parts[f"{x}Temple"].vertices).copy()
            v[:, 2] = hz + (v[:, 2] - hz) * k
            parts[f"{x}Temple"].vertices = v


def normalise_orientation(parts: dict[str, trimesh.Trimesh]) -> None:
    """Sanity: front +Z, temples -Z, left = -X. Raises instead of guessing."""
    lens_z = np.mean([parts[f"{s}Lens"].bounds.mean(axis=0)[2] for s in ("Left", "Right")])
    temple_z = np.mean([parts[f"{s}Temple"].bounds.mean(axis=0)[2] for s in ("Left", "Right")])
    if not temple_z < lens_z:
        raise ValueError("temples must lie behind (-Z of) the lenses")
    if not parts["LeftLens"].bounds.mean(axis=0)[0] < parts["RightLens"].bounds.mean(axis=0)[0]:
        raise ValueError("LeftLens must be at -X")


# --------------------------------------------------------------------------- metadata files
@dataclass
class TemplateSpec:
    name: str
    shape: str                    # backend.models.FrameShape value
    material: str                 # backend.models.FrameMaterial value
    style: str
    bridge_type: str = "keyhole"  # keyhole | straight
    rim_pull_strength: float = 0.45
    temple_curve_angle: float = 28.0
    nose_pad_distance: float = 18.0
    notes: str = ""
    source: str = ""
    target_front_width: float | None = None     # mm; front scaled uniformly to this (engine does not resize width)
    target_temple_length: float | None = None   # mm; temples stretched along Z only, about the hinge
    extra_dims: dict = field(default_factory=dict)


def measure(parts: dict[str, trimesh.Trimesh]) -> dict[str, float]:
    front = np.vstack([parts[p].vertices for p in ("Frame", "LeftRim", "RightRim", "Bridge")])
    ll, rl = parts["LeftLens"], parts["RightLens"]
    temple_len = max(np.ptp(parts[f"{s}Temple"].vertices[:, 2]) for s in ("Left", "Right"))
    rim_t = []
    for s in ("Left", "Right"):
        ok, _, area = aperture_status(parts[f"{s}Rim"], parts[f"{s}Lens"], s)
        if ok:
            r = parts[f"{s}Rim"]
            outer = np.ptp(r.vertices[:, 1]) * np.ptp(r.vertices[:, 0])
            rim_t.append(min(3.0, max(0.8, 0.5 * (np.sqrt(outer) - np.sqrt(area)))))
    return {
        "frame_width": round(float(np.ptp(front[:, 0])), 1),
        "lens_width": round(float(np.mean([np.ptp(ll.vertices[:, 0]), np.ptp(rl.vertices[:, 0])])), 1),
        "lens_height": round(float(np.mean([np.ptp(ll.vertices[:, 1]), np.ptp(rl.vertices[:, 1])])), 1),
        "bridge_width": round(float(2 * min(abs(ll.bounds[1][0]), abs(rl.bounds[0][0]))), 1),
        "temple_length": round(float(temple_len), 1),
        "rim_thickness": round(float(np.median(rim_t)) if rim_t else 1.2, 1),
    }


def write_metadata(spec: TemplateSpec, parts: dict[str, trimesh.Trimesh], templates_dir: Path) -> dict:
    dims = measure(parts)
    full_dims = {
        **dims,
        "temple_curve_angle": spec.temple_curve_angle,
        "nose_pad_distance": spec.nose_pad_distance,
        "nose_pad_angle": 15.0,
        "nose_pad_height": 3.0,
        **spec.extra_dims,
    }
    meta = {
        "name": spec.name,
        "shape": spec.shape,
        "material": spec.material,
        "style": spec.style,
        "glb_file": f"{spec.name}.glb",
        "fallback_glb": f"{spec.name}.glb",
        "dimensions": full_dims,
        "parts": ["Frame", "Bridge", "LeftTemple", "RightTemple", "LeftLens", "RightLens", "LeftRim", "RightRim"],
        "rim_pull_strength": spec.rim_pull_strength,
    }
    if spec.source:
        meta["source_asset"] = spec.source
    if spec.notes:
        meta["notes"] = spec.notes
    (templates_dir / f"{spec.name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    d = json.loads((ROOT / "templates" / "descriptors" / "rectangle_plastic.json").read_text(encoding="utf-8"))
    d["template_name"] = d["template"] = spec.name
    d["style"] = spec.style
    d["bridge_width_mm"] = dims["bridge_width"]
    d["lens_width_mm"] = dims["lens_width"]
    d["lens_height_mm"] = dims["lens_height"]
    d["frame_width_mm"] = dims["frame_width"]
    d["temple_length_mm"] = dims["temple_length"]
    d["bridge"]["type"] = spec.bridge_type
    d["deformation_regions"]["bridge_region"]["type"] = spec.bridge_type
    (templates_dir / "descriptors" / f"{spec.name}.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return meta


def write_registry_entry(spec: TemplateSpec, dims: dict, templates_dir: Path) -> None:
    """Add the template to registry.json with a RELATIVE glb path (no machine-specific paths)."""
    path = templates_dir / "registry.json"
    reg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    base = dict(reg.get("rectangle_plastic", {}))
    base.update(
        name=spec.name,
        style=spec.style,
        glb_path=f"templates/{spec.name}.glb",
        bridge_width_mm=dims["bridge_width"],
        temple_length_mm=dims["temple_length"],
        rim_pull_strength=spec.rim_pull_strength,
        fallback_glb=f"{spec.name}.glb",
        created_at=_dt.date.today().isoformat() + "T00:00:00",
        last_calibrated=None,
    )
    reg[spec.name] = base
    path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")


def finish(
    spec: TemplateSpec,
    parts: dict[str, trimesh.Trimesh],
    out_dir: Path,
    markers: dict[str, np.ndarray] | None = None,
    register: bool = True,
) -> None:
    fit_real_size(parts, spec.target_front_width, spec.target_temple_length)
    centre_x(parts)
    normalise_orientation(parts)
    for s in ("Left", "Right"):
        ok, msg, area = aperture_status(parts[f"{s}Rim"], parts[f"{s}Lens"], s)
        if not ok:
            raise ValueError(f"{spec.name}: {s} rim has no usable aperture: {msg}")
    if markers is None:
        markers = default_markers(parts)
    scene = build_scene(parts, markers)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / f"{spec.name}.glb"
    scene.export(str(glb))
    (out_dir / "descriptors").mkdir(exist_ok=True)
    meta = write_metadata(spec, parts, out_dir)
    if register:
        write_registry_entry(spec, meta["dimensions"], out_dir)
    print(f"wrote {glb}")
    for n in PART_NAMES:
        g = parts[n]
        print(f"  {n:11s} v={len(g.vertices):6d} f={len(g.faces):6d} ext(mm)={np.round(g.extents, 1).tolist()} ctr={np.round(g.centroid, 1).tolist()}")
    print("  dims:", meta["dimensions"])
