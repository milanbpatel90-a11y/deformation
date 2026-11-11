"""Run inside Rhino 8 Python with NewWayfarer_production.3dm open.

Validates GT_002 source naming, records measured part/landmark data and exports
selected runtime geometry. It does not invent a deformation basis.
"""
from pathlib import Path
import json
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc

PARTS = ["Frame","LeftRim","RightRim","Bridge","LeftLens","RightLens","LeftTemple","RightTemple","LeftHinge","RightHinge"]
LANDMARKS = [
 "LM_BridgeCenter","LM_BridgeLeftAttach","LM_BridgeRightAttach",
 "LM_LeftLensCenter","LM_RightLensCenter","LM_LeftLensTop","LM_LeftLensBottom",
 "LM_LeftLensNasal","LM_LeftLensTemporal","LM_RightLensTop","LM_RightLensBottom",
 "LM_RightLensNasal","LM_RightLensTemporal","LM_LeftHinge","LM_RightHinge",
 "LM_LeftHingeAxis","LM_RightHingeAxis","LM_LeftTempleRoot","LM_RightTempleRoot",
 "LM_LeftTempleTip","LM_RightTempleTip"
]

def named_object(name):
    ids = rs.ObjectsByName(name) or []
    if len(ids) != 1:
        raise RuntimeError(f"Expected one object named {name!r}; found {len(ids)}")
    return ids[0]

def bbox(object_id):
    pts = rs.BoundingBox(object_id, view_or_plane=Rhino.Geometry.Plane.WorldXY, in_world_coords=True)
    xs=[p.X for p in pts]; ys=[p.Y for p in pts]; zs=[p.Z for p in pts]
    return {"min":[min(xs),min(ys),min(zs)],"max":[max(xs),max(ys),max(zs)],
            "size":[max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)]}

def landmark_position(object_id):
    obj=sc.doc.Objects.FindId(object_id)
    g=obj.Geometry
    if isinstance(g, Rhino.Geometry.Point):
        p=g.Location
    else:
        p=g.GetBoundingBox(True).Center
    return [p.X,p.Y,p.Z]

def main():
    doc_path=Path(sc.doc.Path)
    if not doc_path.name:
        raise RuntimeError("Save the Rhino document before exporting GT_002.")

    part_ids={n:named_object(n) for n in PARTS}
    lm_ids={n:named_object(n) for n in LANDMARKS}
    report={
      "template_id":"GT_002",
      "source_file":doc_path.name,
      "rhino_version":Rhino.RhinoApp.Version.ToString(),
      "document_unit_system":str(sc.doc.ModelUnitSystem),
      "parts":{n:{"bbox":bbox(i)} for n,i in part_ids.items()},
      "landmarks":{n:{"position":landmark_position(i)} for n,i in lm_ids.items()}
    }

    out_dir=doc_path.parent / "GT_002_runtime"
    out_dir.mkdir(exist_ok=True)
    (out_dir/"rhino_export_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    rs.UnselectAllObjects()
    rs.SelectObjects(list(part_ids.values()))
    glb=out_dir/"template.glb"
    ok=rs.Command(f'_-Export "{glb}" _Enter', echo=True)
    if not ok or not glb.exists():
        raise RuntimeError("GLB export did not produce template.glb; check Rhino glTF exporter options.")
    print(f"GT_002 export complete: {glb}")
    print(f"Measured report: {out_dir/'rhino_export_report.json'}")

if __name__ == "__main__":
    main()
