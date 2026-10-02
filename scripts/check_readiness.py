"""Run asset and physical calibration checks before deployment."""
import json
import os
from pathlib import Path
import numpy as np
from backend.models import Measurements
from backend.deformer.basis_deformer import BasisDeformer
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness
from backend.template_library.compatibility import MeasurementCompatibilityError
from scripts.template_asset_manifest import verify_manifest


def check_gold(root):
    engine = BasisDeformer(root)
    checks = []
    cases = ((135,56,37,18,130.9), (110,40,25,20,120),
             (170,55,50,28,180))
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
    geometry_rejected = False
    try:
        engine.deform(Measurements(frame_width=170,lens_width=40,lens_height=50,
            bridge_width=12,temple_length=120))
    except MeasurementCompatibilityError as exc:
        geometry_rejected = bool(exc.ranges.get('geometry_quality', {}).get(
            'maximum_edge_stretch_ratio', 0) > 3.0)
    return {'passed': all(c['passed'] for c in checks) and rejected and geometry_rejected,
            'invalid_combination_rejected': rejected,
            'unsafe_geometry_rejected_with_ranges': geometry_rejected, 'cases': checks}


def main():
    library = TemplateLibrary()
    report = template_readiness(library)
    try:
        report['asset_delivery'] = verify_manifest()
    except (OSError, ValueError, KeyError) as exc:
        report['asset_delivery'] = {'passed': False, 'error': str(exc)}
        report['ready'] = False
    report['deformation_checks'] = {}
    if 'GT_001' in report['templates']:
        try:
            result = check_gold(Path(library.load('GT_001').glb_path).parents[1])
        except (ValueError, OSError) as exc:
            result = {'passed': False, 'error': str(exc)}
        report['deformation_checks']['GT_001'] = result
        report['ready'] = report['ready'] and result['passed']
    manifest = Path('dataset/real_products/manifest.json')
    product_count = 0
    if manifest.is_file():
        try:
            product_count = len(json.loads(manifest.read_text(encoding='utf-8')).get('products', []))
        except (OSError, json.JSONDecodeError, AttributeError):
            product_count = 0
    production_mode = os.getenv('DEFIRM_PRODUCTION_MODE', '').lower() in {'1', 'true', 'yes'}
    blockers = []
    if not report['ready']:
        blockers.append('Required GT_001 runtime assets and calibrated deformation checks are not ready.')
    if not production_mode:
        blockers.append('DEFIRM_PRODUCTION_MODE=1 is required to put output in the production directory and enforce exact per-component intersection checks.')
    if product_count < 100:
        blockers.append(f'Real-product benchmark has {product_count} labeled products; at least 100 are required for release evidence.')
    if product_count >= 100:
        benchmark_report = Path('output/debug/real_products/report.json')
        try:
            result = json.loads(benchmark_report.read_text(encoding='utf-8'))
            if not result.get('production_validation', {}).get('passed'):
                blockers.append('The latest real-product benchmark did not pass its configured release criteria.')
        except (OSError, json.JSONDecodeError, AttributeError):
            blockers.append('Run the real-product benchmark and save its report before release.')
    release_file = Path('dataset/real_products/release_acceptance.json')
    try:
        release = json.loads(release_file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        release = {}
    if not (release.get('browser_vto_verified') is True and release.get('mobile_vto_verified') is True
            and release.get('reviewed_by') and release.get('tested_devices')):
        blockers.append('Record browser and mobile VTO acceptance in dataset/real_products/release_acceptance.json.')
    report['production_ready'] = not blockers
    report['production_blockers'] = blockers
    report['real_product_benchmark_count'] = product_count
    print(json.dumps(report, indent=2))
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
