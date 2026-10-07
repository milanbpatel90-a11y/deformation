"""Run the full pipeline on the genuine multi-angle fixture and inspect the result."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.pipeline.video_pipeline import VideoDeformationPipeline
from tests.multi_angle_fixture import multi_angle_video
from tests.video_fixture import scratch_dir

work = scratch_dir("_multi_angle_run")
video = multi_angle_video(work / "orbit.mp4", repeats=4, fps=20.0)
print("clip:", video, f"{video.stat().st_size/1024:.0f} KB")

result = VideoDeformationPipeline().run_from_video(
    video, work / "orbit.glb", preview_dir=work / "previews"
)
video_section = result["video"]

print("\n=== selection ===")
sel = video_section["selection"]
print("selected:", sel["selected_count"], "of gate-accepted", sel["candidates"],
      "| pool", sel["pool_candidates"])
print("view distribution:", video_section["view_distribution"])
print("all front?", set(video_section["view_distribution"]) == {"front"})

print("\n=== per-view classification (evidence) ===")
for entry in video_section["per_view"]:
    ev = entry["evidence"]
    print(f"  frame {entry['frame_index']:3d} {entry['view']:<26s} conf={entry['view_confidence']:.2f} "
          f"yaw={entry['yaw_deg']:6.1f} temple_vis={ev.get('temple_visibility', 0):.3f} "
          f"fill={ev.get('bbox_fill', 0):.3f} skew={ev.get('lateral_skew', 0):+.3f}")

print("\n=== per-dimension provenance ===")
for name, entries in video_section["per_dimension_provenance"].items():
    measured = [e for e in entries if e["source"] != "insufficient_evidence"]
    sources = Counter(e["source"] for e in measured)
    values = [e["value_mm"] for e in measured]
    span = f"{min(values):.1f}-{max(values):.1f} mm" if values else "INSUFFICIENT"
    print(f"  {name:<20s} {span:>18s}  {dict(sources) if sources else 'no evidence'}")

print("\n=== fusion ===")
for name, report in video_section["fusion"]["dimensions"].items():
    print(f"  {name:<20s} {report['value']:8.2f} {report['unit']:<4s} "
          f"src={report['source']:<22s} kept={report['kept']}/{report['contributors']} "
          f"spread={report['spread']:.3f} evidence={report.get('evidence_mix')}")

print("\n=== scale ===")
print(" ", json.dumps(video_section["scale"]))
print("\n=== confidence ===")
print(" ", json.dumps(video_section["confidence"], indent=2)[:900])
print("\n=== validation ===")
print(" ", json.dumps(video_section["validation"]))

print("\n=== fused measurements ===")
for key in ("frame_width", "lens_width", "lens_height", "bridge_width",
            "temple_length", "rim_thickness", "temple_curve_angle"):
    print(f"  {key:<20s} {result['measurements'][key]}")
print("template:", result["template"], "| acceptance:", result["acceptance"].get("status"))
print("manifest:", result["video_manifest_json"])

manifest = json.loads(Path(result["video_manifest_json"]).read_text(encoding="utf-8"))
print("\nmanifest keys:", sorted(manifest.keys()))
