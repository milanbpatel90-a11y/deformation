"""Run asset and physical calibration checks before deployment."""
import json
from pathlib import Path
import numpy as np
from backend.models import Measurements
from backend.deformer.basis_deformer import BasisDeformer
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness
from backend.template_library.compatibility import MeasurementCompatibilityError


def check_gold(root):
    engine = BasisDeformer(root)
    checks = []
    cases = ((135,56,37,18,130.9), (110,40,25,12,120),
             (170,69.8,50,28,180), (170,40,50,12,120))
    for fw,lw,lh,bw,tl in cases:
        scene,quality = engine.deform(Measurements(frame_width=fw,lens_width=lw,
            lens_height=lh,bridge_width=bw,temple_length=tl))
        frame,left,right = (scene.geometry[n].vertices for n in ('Frame','LeftLens','RightLens'))
        measured = np.array([np.ptp(frame[:,0]),np.ptp(left[:,0]),np.ptp(left[:,2]),
                             left[:,0].min()-right[:,0].max()])
        error = float(np.max(np.abs(measured-[fw,lw,lh,bw])))
        checks.append({'requested_mm': [fw,lw,lh,bw,tl], 'maximum_error_mm': error,
                       'minimum_jacobian': engine.regions.min_jacobian,
                       'passed': bool(error<=0.5 and quality.passed and engine.regions.min_jacobian>0)})
    rejected = False
    try:
        engine.deform(Measurements(frame_width=135,lens_width=60,lens_height=37,
            bridge_width=20,temple_length=155))
    except MeasurementCompatibilityError:
        rejected = True
    return {'passed': all(c['passed'] for c in checks) and rejected,
            'invalid_combination_rejected': rejected, 'cases': checks}


def main():
    library = TemplateLibrary()
    report = template_readiness(library)
    report['deformation_checks'] = {}
    if 'GT_001' in report['templates']:
        try:
            result = check_gold(Path(library.load('GT_001').glb_path).parents[1])
        except (ValueError, OSError) as exc:
            result = {'passed': False, 'error': str(exc)}
        report['deformation_checks']['GT_001'] = result
        report['ready'] = report['ready'] and result['passed']
    print(json.dumps(report, indent=2))
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
