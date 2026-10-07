"""Inspect the measurement estimate for the multi-angle orbit, stage by stage."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.pipeline.video_pipeline import VideoDeformationPipeline
from backend.video.frame_extractor import FrameExtractor
from backend.video.quality_gate import FrameQualityGate
from backend.video.selection import FrameSelector
from tests.multi_angle_fixture import multi_angle_video
from tests.video_fixture import scratch_dir

work = scratch_dir("_multi_angle_run")
video = multi_angle_video(work / "orbit.mp4", repeats=4, fps=20.0)

extraction = FrameExtractor(target_fps=6.0).extract(video)
gate = FrameQualityGate()
qualities = gate.evaluate_all([f.image for f in extraction.frames])
print("sampled:", len(extraction.frames), "gate accepted:", sum(1 for q in qualities if q.accepted))
print("gate rejects:", dict(Counter(r.split(" (")[0] for q in qualities if not q.accepted for r in q.reasons)))

pool = FrameSelector().preselect(extraction.frames, qualities, 20)
print("pool:", [item.frame.index for item in pool.selected])

estimate = VideoDeformationPipeline().estimate_from_video(video)
print("\nclassification distribution:", estimate["classification_distribution"])
print("orbit:", json.dumps(estimate["orbit"].to_dict()))
print("\nselected frames:")
for entry in estimate["selected_views"]:
    print(f"  idx {entry['index']:3d} {entry['view']:<26s} conf={entry['view_confidence']:.2f} "
          f"yaw={entry.get('yaw_deg', 0):6.1f} q={entry['quality'].score:.3f}")

print("\nper-dimension provenance:")
for name, entries in estimate["orbit_measurement"].provenance().items():
    measured = [e for e in entries if e["source"] != "insufficient_evidence"]
    srcs = Counter(e["source"] for e in measured)
    vals = [e["value_mm"] for e in measured]
    print(f"  {name:<20s} {f'{min(vals):.1f}-{max(vals):.1f}' if vals else 'INSUFFICIENT':>16s}  {dict(srcs)}")
    for e in measured[:3]:
        print(f"       idx {e['frame_index']:3d} {e['view']:<26s} yaw={e['yaw_deg']:6.1f} "
              f"{e['source']:<24s} px={e['value_px']:7.1f} mm={e['value_mm']:7.2f}")

print("\nscale:", json.dumps(estimate["scale"]))
print("\nfused:", {k: estimate["measurements"].model_dump()[k] for k in
                   ("frame_width", "lens_width", "lens_height", "bridge_width",
                    "temple_length", "rim_thickness", "temple_curve_angle")})
print("\nfusion sources:", {k: v["source"] for k, v in estimate["fusion"].to_dict()["dimensions"].items()})
print("confidence:", estimate["confidence"]["level"], estimate["confidence"]["caps"])
