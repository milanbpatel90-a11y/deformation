"""Run with python -m scripts.audit_glb FILE --expected-width-mm 135."""
import argparse
import json
from pathlib import Path
from backend.exporter.validation import inspect_glb

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('glb')
    parser.add_argument('--expected-width-mm',type=float)
    parser.add_argument('--tolerance-mm',type=float,default=0.5)
    parser.add_argument('--output')
    args=parser.parse_args()
    report=inspect_glb(args.glb,args.expected_width_mm,args.tolerance_mm)
    text=json.dumps(report,indent=2)
    if args.output:
        Path(args.output).write_text(text,encoding='utf-8')
    else:
        print(text)
    raise SystemExit(1 if report['dimension_passed'] is False else 0)
