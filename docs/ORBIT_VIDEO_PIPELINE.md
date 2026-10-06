# Orbit video pipeline

Turn an 8-12 second 360 degree orbit clip of an eyewear product into a
validated, deformable GLB. This path complements the still-image endpoints: it
replaces a hand-picked photo with many automatically chosen views, and it
replaces averaging with robust statistics.

```
upload video
  -> incremental time-grid sampling     backend/video/frame_extractor.py
  -> quality gate (blur/glare/exposure) backend/video/quality_gate.py
  -> candidate frames (gate survivors)
  -> duplicate removal + coverage       backend/video/selection.py
     -> bounded candidate pool (<= 2x max views)
  -> 1-class YOLO eyewear segmentation
  -> eyewear visibility gate            backend/video/quality_gate.py
  -> view classification (front / left-right perspective / side / top)
  -> coverage-aware selection of 5-10   backend/video/selection.py
  -> measure each view                  backend/measurement/extractor.py
  -> weighted median + MAD fusion       backend/fusion/robust_fuser.py
  -> template selection
  -> existing deformation engine        backend/pipeline/deformation_pipeline.py
  -> GLB + independent GLB validation
  -> manifest + selected frames + contact sheet
```

Orchestration lives in `backend/pipeline/video_pipeline.py`, which subclasses
`DeformationPipeline` and reuses the matcher, deformers, material pass and
serialized-GLB validation unchanged. This is the **only** video implementation in
the repository; see `docs/VIDEO_PIPELINE_CONSOLIDATION.md` for the four rival
implementations that were reconciled into it and what was absorbed from each.

## Using it

### API

`POST /api/deform/video` (multipart)

| field | required | meaning |
|---|---|---|
| `video` | yes | the orbit clip; `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, up to 200 MiB |
| `color` | no | frame colour, `#rrggbb` |
| `template` | no | force a template instead of auto-selecting |
| `measurements` | no | manual `Measurements` JSON to override the fused sizes |
| `automatic_appearance` | no | detect colour/material/shape from the anchor view (default true) |
| `min_views` | no | fewest views to accept, default 5 |
| `target_views` | no | preferred view count, default 8 |
| `max_views` | no | most views to fuse, default 10 |
| `target_fps` | no | decode sampling rate, default 6.0 |
| `reference_width_mm` | no | known real frame width (20-250 mm) used to scale the estimate |

The response matches the other deform endpoints and adds `selected_views`,
`confidence`, `contact_sheet_url`, `video_manifest_url`, and a `video` section:
`decode`, `gate`, `pool`, `selection`, `per_view`, `view_distribution`, `model`,
`anchor`, `fusion`, `unusable_frames`, `measurement_source`, `scale`,
`confidence`, `rear_available`, `previews` (`frame_urls`, `grid_url`) and
`timings`. Rejection paths return 4xx with an actionable message rather than a
generic failure, for example:

```json
{"detail": "Orbit video is 0.4s long; record at least 2s (8-12s recommended)."}
{"detail": "Only 1 frame(s) could be decoded from the upload. The file does not appear to be a video, or its codec is unsupported. Re-export it as an H.264 MP4."}
{"detail": "Only 3 frame(s) contained usable eyewear; at least 5 are required. Film a slower, steadier 360 degree orbit with the whole frame inside the picture and no occluding hands."}
{"detail": "Selected views cover only 2 of 8 angular sectors. The usable frames come from one part of the video and cannot describe a full orbit. Record a complete 360 degree turn."}
```

Selected frames are written as JPEGs under `output/development/<job_id>_frames/`
and served from `GET /api/output/frames/{job_id}/{filename}` (strictly confined
to that directory). A `selected_frames.jpg` contact sheet is included, and the
orbit manifest is served from `GET /api/output/<job_id>.video.json` through the
same confined route as the GLB. The uploaded clip itself is deleted as soon as
the job finishes, including on failure.

### Viewer

`/viewer/` gains a **360° orbit video** card under the image uploads. Selecting a
clip shows an inline preview and warns when the duration is outside 8-12 s,
client-side only; the server enforces the real limits. A clip supersedes the
still-image inputs. After a run, the **Orbit views used** panel reports the
selected/candidate/decoded counts, rejected and duplicate counts, the view
distribution, angular spread, confidence level with its two consistency
components, the chosen template, the scale assumption and the runtime — then a
labelled thumbnail of every selected view (frame number, timestamp, view class,
quality score) plus links to the contact sheet and the orbit manifest. The GLB
loads into the existing 3D viewer, and webcam try-on is untouched.


## Design decisions

### The gate runs before segmentation; coverage is checked after

The quality gate is mask-free and cheap, so it screens every sampled frame with
a few OpenCV reductions. Duplicate removal and coverage ranking then narrow the
survivors to a bounded candidate pool (twice the maximum final view count), and
segmentation runs on that pool alone -- never on all ~60-70 sampled frames, and
never on the whole clip. Measured on a 10 s / 20 fps clip:

| stage | long_edge=1280 | long_edge=640 |
|---|---|---|
| decode (67 sampled) | 0.32 s | 0.42 s |
| quality gate (67) | 2.16 s | 1.64 s |
| selection | 0.43 s | 0.13 s |
| segmentation (20 pooled, 8 kept) | 4.01 s | 2.81 s |

A full run including deformation, materials and GLB validation takes about
**11-14 s** on this machine (13.6 s measured through the HTTP endpoint).
Per-frame YOLO latency is ~208 ms at 1280x720 and ~166 ms at 640x360, so frame
resolution is a usable throughput dial.

Semantics are checked at two points, deliberately. Subject *coverage* is
approximated before segmentation, from a per-pixel temporal median across the
clip: the median estimates the static backdrop and each frame's difference from
it is its moving subject. That is what lets redundancy and angular spread be
judged without running the model. The authoritative eyewear-visibility check --
is the subject present, cut off by the frame edge, or broken into fragments --
runs after segmentation on the real masks, because only a mask can answer those
questions.

### Selection maximises the minimum pairwise spread

Frames are screened, then soft frames are dropped relative to the clip's own
median sharpness (so a uniformly soft clip is not starved), then near-identical
temporal neighbours are collapsed, then farthest-point sampling picks the frames
that are furthest apart in silhouette-descriptor space, seeded by the
highest-quality frame and broken by quality. Near-duplicates are clustered first
and each cluster survives as its **best** frame, not its earliest, so a run of
ten near-identical frames contributes one view. On the synthetic 10 s orbit this
selects 8 views from 67 sampled frames with a mean nearest-neighbour descriptor
distance of 0.146, occupying 5 of 8 angular sectors and spanning the whole clip.

### Fusion is a weighted median with MAD screening

`backend/fusion/robust_fuser.py` fuses each dimension only from the views that
can actually see it:

| view | frame | lens w | lens h | bridge | rim | temple | curve |
|---|---|---|---|---|---|---|---|
| front | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0 | 0 |
| left_front_perspective | 0.55 | 0.45 | 0.45 | 0.45 | 0.20 | 0.35 | 0.30 |
| right_front_perspective | 0.55 | 0.45 | 0.45 | 0.45 | 0.20 | 0.35 | 0.30 |
| side | 0 | 0 | 0 | 0 | 0.30 | 1.00 | 1.00 |
| top | 0.50 | 0.30 | 0.10 | 0.40 | 1.00 | 0.40 | 0.25 |
| rear | *reserved, never emitted* | | | | | | |

A weight of 0 removes the view entirely, which is the point: the bounding box of
a profile is the temple length, not the frame width, so a side frame must not
contribute a frame width. Each view's weight is additionally scaled by its
quality score and by subject coverage. An unrecognised view label contributes
nothing and is reported in `warnings` rather than being silently dropped.

Per dimension the fuser takes a weighted median, computes the MAD, rejects
values beyond `mad_k * 1.4826 * MAD` (with a relative floor so a unanimous
cluster is not split by float jitter), then takes a weighted median of the
survivors. It reports `median`, `mad`, `robust_sigma`, `spread`, `agreement`,
`contributors`, `kept` and every rejected value with its deviation, so a
disagreement is visible instead of hidden behind one number. On synthetic data
with two low outliers the mean is dragged 13.8 mm while the robust estimate
moves 0.0 mm.

### Confidence is derived from the evidence

`FusionResult.confidence()` returns the documented object:

```json
{
  "level": "medium",
  "score": 0.68,
  "selected_views": 8,
  "measurement_consistency": 0.70,
  "quality_consistency": 0.65,
  "scale_assumption": "reference_width",
  "notes": ["estimated without direct evidence: temple_length"]
}
```

* `measurement_consistency` is mean cross-view agreement over dimensions that
  were *measured*, scaled by the fraction of dimensions that had direct evidence
  at all. A dimension filled in by a fallback carries no evidence, so it lowers
  the score instead of being hidden, and it is named in `notes`.
* `quality_consistency` is mean view *trust* -- frame quality folded with how
  much of the frame the eyewear occupies -- scaled by how close the view count is
  to the preferred 8. Per-view trust is in the manifest, so the aggregate is
  auditable.
* `high` additionally requires at least 8 views. Perfect agreement between 7
  views is not the same claim as between 8, and the level is capped at `medium`
  with a note saying why.

The level therefore falls when views disagree, when dimensions were estimated,
when frames are low quality or the subject is small, and when too few views
contributed. It does not rise merely because the pipeline completed.


## Threshold calibration

Gate thresholds were fitted against `test_images/*.jpg` and synthetic
degradations of them. Frames accepted out of 6 test photos:

| variant | accepted |
|---|---|
| sharp | 6/6 |
| Gaussian blur sigma 1.0 | 6/6 |
| Gaussian blur sigma 2.0 | 0/6 |
| Gaussian blur sigma 3.5 | 0/6 |
| horizontal motion blur, 15 px | 6/6 |
| horizontal motion blur, 31 px | 3/6 |
| glare 2% of frame | 6/6 |
| glare 6% of frame | 0/6 |
| glare 15% of frame | 0/6 |
| exposure x0.30 | 1/6 |
| exposure x0.15 | 0/6 |
| exposure x1.60 | 0/6 |

Two calibration errors were found and corrected during development rather than
shipped:

* `MIN_EDGE_DENSITY` rejected 4 of 6 *sharp* photos (their edge density was
  0.0022 against a 0.0030 floor). Edge density is now reported but does not gate.
* `MIN_SUBJECT_COVERAGE` was set to 0.015, which discarded distant-but-valid
  orbit frames measuring ~0.005 and left only 3 of 24 pooled frames. It is now
  0.002: only a genuinely empty mask is treated as "no eyewear", and partial
  trust is expressed as a weight instead.

Mild blur (sigma 1.0) and a small 2% highlight are deliberately accepted, because
orbit video is softer than a studio still and these do not change a measurement.
Extreme directional smear is caught separately by `gradient_anisotropy`, since a
motion-blurred frame can keep a plausible Laplacian variance while losing all
detail along one axis.

## Limitations

**Absolute scale is still assumed.** `MeasurementExtractor` calibrates on a fixed
135 mm frame-width reference, so `frame_width` is *identical in every view by
construction* (`mm_per_px = 135 / bbox_width`, then `frame_width = bbox_width *
mm_per_px`). Lens and bridge values are therefore proportions of that 135 mm
assumption, and temple length and rim thickness come from their own fixed
references. The orbit path makes these estimates more robust and consistent -- it
cannot make them true millimetres. A scale reference (a known dimension, a
calibration target, or a depth sensor) is required for that. This matches the
caveat already recorded in `PRODUCTION_READINESS.md`.

**`rear` is not claimed, and the report says so.** The intended front/rear cue was
lens-aperture visibility, but the one-class model emits a *solid filled*
silhouette: `aperture_openness` measured 0.000 and `hole_count` 0 on all 18 images
in `test_images/`. With no holes there is no signal, so `rear` is not in the
label vocabulary at all, `ViewClassifier.REAR_AVAILABLE` is `False`, and every
classification carries `rear_available: 0.0` plus a `rear_available: false` flag
in the manifest. An earlier opt-in outer-edge-thickness heuristic was removed
rather than shipped: on those same 18 front-facing photos edge thickness spans
0.37-0.88, which overlaps any plausible rear-view value, so no threshold in that
range is defensible. Recovering `rear` needs labelled rear-view orbit data, or a
part-level mask that preserves the apertures.

**Left/right perspective is a convention, not a pose estimate.** The half of the
silhouette with more mask area is treated as the side turned towards the camera,
because the nearer lens and rim subtend more area. `lateral_skew` is reported and
the `PERSPECTIVE_SKEW` threshold is exposed so the rule can be re-thresholded
against labelled data. It is a coarse cue for view weighting, not a measured
yaw angle.

**`side` and `top` are exercised by constructed masks, not real data.** The
end-to-end fixture is built by foreshortening a real photo, which produces
credible frontal-hemisphere frames but never a true profile or plan view, so the
synthetic end-to-end runs report only `front`. `side` and `top` are therefore
covered by deterministic mask tests rather than by a real orbit clip. Their
thresholds, and the fusion weights that depend on them, would benefit from real
turntable data.

**Occlusion is approximated, not detected.** A binary eyewear mask cannot say
*what* is in front of the glasses, only that the silhouette has been cut off by
the frame edge or has come apart into comparable fragments. The visibility gate
uses exactly those signals and does not pretend to score occlusion directly.

**Preview directories accumulate.** Each orbit job leaves up to 11 small JPEGs
(~300 KB) under `output/development/<job_id>_frames/`. There is no retention
policy yet; add one before public deployment.

## Related fixes

Two pre-existing defects surfaced while building this path and were fixed,
because the orbit pipeline cannot work without them:

1. **The trained 1-class model was shadowed.** `models/best.pt` holds a six-class
   part model (`eyewear_rim`, `eyewear_temple`, `bridge`, `left_lens`,
   `right_lens`, `nose_pad`) that returns **zero detections on all 18** test
   images, and it was the first candidate `GlassesSegmenter._find_model()` found.
   The working one-class `eyewear` weights at
   `runs/segment/train/weights/best.pt` detect on **18/18** (confidence
   0.26-0.99) and are what the repo's own setup guide means by
   `cp runs/segment/train/weights/best.pt models/best.pt`. The search order now
   prefers the one-class weights, and a model that detects nothing falls back to
   the classical mask with the fallback reported in `model_type` instead of
   silently yielding an empty mask that the measurement stage would replace with
   default sizes.

2. **Automatic measurements were structurally infeasible.** The extractor sized
   the two lens widths and the bridge from the *entire* frame width, so
   `2*lens_width + bridge_width == frame_width` exactly, leaving nothing for the
   rim thickness that `validate_combination` requires
   (`frame_width >= 2*lens + bridge + 2*rim`). Every automatic, non-manual
   measurement was rejected by the deformer. Rim thickness is now resolved first
   and the lenses and bridge are sized from the width the rims leave, with a
   rounding margin and a final feasibility repair. The image endpoints never
   exposed this because they require manual measurements, and
   `test_production_contracts` monkeypatches the extractor away -- orbit video is
   the first path to feed automatic measurements into the deformer.

## Tests

`tests/test_video_pipeline.py` (66 tests) covers frame extraction and its bounds,
the quality gate against calibrated degradations, eyewear visibility (empty mask,
edge cropping, comparable fragments, tolerant specks), the 5-10 view contract
(never more than the maximum, never more than available, fewer than the minimum
is a clear error, deterministic across runs), duplicate removal keeping the best
representative, one-sector-of-the-clip rejection, bounded pre-selection, the
weighted median on hand-computed values, outlier rejection versus the mean,
per-dimension view routing, evidence-based confidence, the orbit view labels,
single-class model selection, the automatic-measurement feasibility regression,
and five end-to-end runs through the real model and deformers.

The end-to-end contract tests do not mock the deformation logic: they run the
real `DeformationPipeline`, the real deformers and the real serialized-GLB
validation, and assert the resulting artifact.

`tests/video_fixture.py` synthesises clips from a real eyewear photo, because a
drawn shape would never be segmented by the real model and an end-to-end test
built from one would prove nothing. It also documents why clips must be written
to a project-local directory: OpenCV's path-based file APIs fail on Windows 8.3
short paths such as `PETPOO~1`, which is what the system temp directory resolves
to on some machines.

```powershell
python -m pytest tests/test_video_pipeline.py -q
```
