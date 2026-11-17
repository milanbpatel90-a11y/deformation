"""Template acceptance gate: every template must survive the real DeformationPipeline.

Checks per template (exit code 1 if any FAIL):
  1. identity  - requesting the template's own dimensions returns it (max vertex move <= 4 mm,
                 frame width / lens height / temple length within 1 mm), no fallback path
  2. fit matrix - narrow / medium / wide faces produce finite, non-degenerate output whose
                 frame width and lens size track the request within tolerance
  3. reload     - exported GLB loads, keeps every part, has no NaN vertices
Usage: python tools/validate_template_pipeline.py [template ...]   (default: all with a descriptor)
"""
from __future__ import annotations
import json, sys, tempfile, warnings
from pathlib import Path
import numpy as np, trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.models import Measurements  # noqa: E402
from backend.pipeline.deformation_pipeline import DeformationPipeline  # noqa: E402

TEMPLATES = Path("templates")
ext = lambda m: m.bounds[1] - m.bounds[0]


def world(path):
    scene = trimesh.load(path, force="scene", process=False)
    out = {}
    for n in scene.graph.nodes_geometry:
        T, g = scene.graph[n]
        m = scene.geometry[g].copy(); m.apply_transform(T); out[n] = m
    return out


def run(pipe, name, dims, out):
    m = Measurements(frame_width=dims["frame_width"], lens_width=dims["lens_width"], lens_height=dims["lens_height"],
                     bridge_width=dims["bridge_width"], temple_length=dims["temple_length"],
                     rim_thickness=dims.get("rim_thickness", 1.2), temple_curve_angle=dims.get("temple_curve_angle", 8.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); pipe.run_from_measurements(m, str(out), template_name=name)
    return any("fallback" in str(x.message).lower() for x in w)


def check(pipe, name, tmp):
    fails = []
    d = json.load(open(TEMPLATES / f"{name}.json"))["dimensions"]
    src = world(TEMPLATES / f"{name}.glb")
    out = tmp / f"{name}_id.glb"
    if run(pipe, name, d, out): fails.append("identity used fallback path")
    dst = world(out)
    missing = set(src) - set(dst)
    if missing: fails.append(f"parts lost: {sorted(missing)}")
    mv = max(float(np.linalg.norm(src[k].vertices - dst[k].vertices, axis=1).max())
             for k in src if k in dst and len(src[k].vertices) == len(dst[k].vertices))
    if mv > 4.0: fails.append(f"identity max vertex move {mv:.1f} mm > 4")
    for part, ax, key in (("Frame", 0, "frame_width"), ("LeftLens", 1, "lens_height"), ("LeftTemple", 2, "temple_length")):
        if part in src and part in dst and abs(ext(src[part])[ax] - ext(dst[part])[ax]) > 1.0:
            fails.append(f"identity {part} extent {ext(src[part])[ax]:.1f}->{ext(dst[part])[ax]:.1f}")
    for label, fw, lw, lh, bw in (("narrow", .93, .93, .95, .9), ("medium", 1.0, 1.0, 1.0, 1.0), ("wide", 1.07, 1.07, 1.05, 1.15)):
        req = dict(d, frame_width=d["frame_width"] * fw, lens_width=d["lens_width"] * lw,
                   lens_height=d["lens_height"] * lh, bridge_width=d["bridge_width"] * bw)
        o = tmp / f"{name}_{label}.glb"
        run(pipe, name, req, o)
        g = world(o)
        if any(not np.isfinite(m.vertices).all() for m in g.values()): fails.append(f"{label}: NaN vertices"); continue
        if set(g) != set(src): fails.append(f"{label}: part set changed")
        got = ext(g["Frame"])[0]
        if abs(got - req["frame_width"]) > 0.06 * req["frame_width"]:
            fails.append(f"{label}: frame width {got:.1f} vs requested {req['frame_width']:.1f}")
        gl = ext(g["LeftLens"])
        if abs(gl[1] - req["lens_height"]) > 0.12 * req["lens_height"] + 1:
            fails.append(f"{label}: lens height {gl[1]:.1f} vs requested {req['lens_height']:.1f}")
        if gl[0] > 1.25 * req["lens_width"]:
            fails.append(f"{label}: lens width {gl[0]:.1f} inflated vs requested {req['lens_width']:.1f}")
    return fails


def main():
    names = sys.argv[1:] or sorted(p.stem for p in (TEMPLATES / "descriptors").glob("*.json") if (TEMPLATES / f"{p.stem}.glb").exists())
    pipe = DeformationPipeline(templates_dir=TEMPLATES)
    bad = 0
    with tempfile.TemporaryDirectory() as t:
        for n in names:
            try: f = check(pipe, n, Path(t))
            except Exception as e: f = [f"exception: {type(e).__name__}: {e}"]
            print(f"{'PASS' if not f else 'FAIL'}  {n}"); [print("      -", x) for x in f]; bad += bool(f)
    sys.exit(1 if bad else 0)

main()
