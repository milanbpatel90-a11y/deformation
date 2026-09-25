"""Reproduce production-asset export, measured audit and deformation sweep."""
import hashlib
import json
from pathlib import Path
import numpy as np
from backend.deformer.basis_deformer import BasisDeformer
from backend.exporter.glb_exporter import GLBExporter
from backend.exporter.geometry import dimensions, world_vertices
from backend.exporter.validation import inspect_glb
from backend.materials.pbr import apply_materials
from backend.models import Measurements
from backend.template_library.compatibility import MeasurementCompatibilityError, validate_combination


def run():
    root=Path('assets/templates/GT_001')
    output=Path('output/glb-audit')
    output.mkdir(parents=True,exist_ok=True)
    originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (root/'geometry/template.glb',root/'deformation/basis.npz')}
    engine=BasisDeformer(root)
    values=dict(frame_width=135,lens_width=56,lens_height=37,bridge_width=18,
                temple_length=130.9,rim_thickness=1.2,color='#d0b7b1')
    results=[]
    # Standard input and independent low/high changes, using the real basis.
    cases=[('standard',{})]+[(f'{p["name"]}-{side}',{p['name']:p[side]})
         for p in engine.parameters for side in ('min','max')]
    for name,overrides in cases:
        m=Measurements(**(values | overrides))
        try:
            validate_combination(m)
        except MeasurementCompatibilityError:
            try:
                engine.deform(m)
            except MeasurementCompatibilityError:
                results.append({'case':name,'valid_input':False,'rejected':True,'requested_mm':m.model_dump(mode='json')})
                continue
            raise AssertionError('Engine accepted invalid coupled measurements')
        scene,quality=engine.deform(m)
        entry={'case':name,'valid_input':True,'requested_mm':m.model_dump(mode='json'),
               'measured_mm':dimensions(scene),'quality':quality.to_dict()}
        # Signed lens gap is separate from nominal template bridge control.
        left=world_vertices(scene,'LeftLens');right=world_vertices(scene,'RightLens')
        entry['lens_box_gap_mm']=float(left[:,0].min()-right[:,0].max())
        if name=='standard':
            path=output/'standard.glb'
            GLBExporter().export(apply_materials(scene,m),path,m,'GT_001')
            entry['export']=inspect_glb(path,m.frame_width,0.01)
        results.append(entry)
    assert originals=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in originals}
    report={'source_sha256':originals,'cases':results,
            'all_cases_production_ready':all((x.get('rejected',False) if not x['valid_input'] else
                 x['quality']['passed'] and x['lens_box_gap_mm']>0) for x in results)}
    (output/'regression.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(results),'source_assets_unchanged':True,
         'all_cases_production_ready':report['all_cases_production_ready'],
         'standard':{k:results[0]['export'][k] for k in ('bytes','vertices','triangles','material_count','frame_width_world','dimension_passed')}},indent=2))
    # Inspection can succeed while dimensional readiness is false; make that
    # distinction explicit in the report and exit status for CI.
    return 0 if report['all_cases_production_ready'] else 2

if __name__=='__main__':
    raise SystemExit(run())
