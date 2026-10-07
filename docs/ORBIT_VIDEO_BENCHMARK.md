# Orbit video benchmark

How to record, annotate and score **real 360-degree eyewear orbit videos**
against the orbit deformation pipeline
(`backend/pipeline/video_pipeline.py::VideoDeformationPipeline`), using
`scripts/benchmark_orbit_videos.py`.

---

> ## Status: NOT RUN — zero real annotated videos exist in this repository
>
> The repository currently contains **zero real annotated orbit videos**.
> `dataset/orbit_benchmark/` holds only a README and
> `annotations/EXAMPLE.schema.json`.
>
> **No accuracy claim is supported by this repository, and none is made.** Every
> number the harness would produce today is `n/a`, and running it against the
> empty/absent dataset exits with code `3` and `status = "not_run"` — never `0`.
>
> There is no production accuracy figure, no "validated" badge, and no measured
> error rate for the orbit pipeline. Anything that looks like one is either the
> single-image benchmark in `scripts/benchmark_real_products.py` (a different
> capture mode, also not a production guarantee) or an estimate from synthetic
> fixtures, which prove plumbing and nothing about real products.

---

## Why this exists

The orbit pipeline's measurement half makes claims that only real data can
settle: that five to ten automatically selected views beat one front photo, that
weighted-median fusion is more accurate than a single frame, that the six
dimensions land within a stated tolerance, and that template selection is
correct across frame families. Synthetic clips built by foreshortening a product
photo (see `tests/video_fixture.py`) are the right tool for regression-testing
plumbing, but they cannot answer any of those questions: they are generated from
the same photo distribution the models were tuned against, they contain no
ground-truth millimetres, and they have no real material, thickness or rim
geometry.

This harness is the instrument for answering them — once real data exists.

## What the harness measures

For every `<name>.<video>` + `<name>.json` pair in the dataset directory, at each
requested view count (default **5, 8 and 10**):

| Group | Reported |
| --- | --- |
| Measurement accuracy | MAE, RMSE and max absolute error per dimension, plus pooled totals, **only where the annotation supplies a real measured value** |
| Template accuracy | Selected template vs the annotation's `category`, as a match/mismatch count and an accuracy rate, **only for clips that declare a category** |
| View selection | Number of views actually selected, and how it compares to the requested target |
| View-class diversity | The distribution of view labels (`front`, `left_front_perspective`, `right_front_perspective`, `side`, `top`) and its normalised entropy — an even spread scores higher than all views landing in one class |
| Confidence | The pipeline's own confidence level (`low`/`medium`/`high`) and score, and the distribution across clips |
| Acceptance | The exported GLB's validation status, and the share of runs that reported `PASS` |
| Baseline | The same measurement error from the still-image pipeline run on the annotated front frame, and the delta (video MAE − baseline MAE) |
| Cost | Per-stage timings from the `video.timings` section |

Everything is written to `report.json` (machine-readable) and `SUMMARY.md`
(markdown) in the output directory (default `output/benchmark/orbit`), and a
console summary table is printed.

## Annotation schema

Only `front_frame` and `measurements` are required for a scored comparison.
Everything else is optional. The authoritative table lives in
`dataset/orbit_benchmark/README.md`; the machine-readable template is
`dataset/orbit_benchmark/annotations/EXAMPLE.schema.json` (null placeholders, no
real data).

| Field | Required | Notes |
| --- | --- | --- |
| `front_frame` | **yes** | Zero-based index of the square-on front frame; also the source of the single-image baseline |
| `measurements` | **yes** (≥1 dimension) | `frame_width`, `lens_width`, `lens_height`, `bridge_width`, `temple_length`, `rim_thickness`, in mm |
| `category` | optional | Metal / acetate / rimless / thick / thin / rectangular / round / cat_eye / wayfarer / aviator. Needed for template accuracy |
| `product_id`, `frame_count`, `fps`, `side_frame`, `top_frame`, `usable_frames`, `expected_selected_views`, `lighting`, `background` | optional | Recorded, sliced and reported; never fabricated if absent |

**Omitted is not zero.** A dimension the annotator did not supply is excluded
from every error metric. It is never scored as a zero-error dimension, and no
value is imputed from another dimension.

## Recording a clip

Real footage only — no renders, no synthetic orbits, no AI-generated video.

1. **Mount** the frame on a stand or turntable at roughly eye height.
2. **Frame it** so the whole product stays inside the picture at every angle,
   including both temple tips.
3. **Orbit slowly**: one revolution in 15–25 seconds, roughly constant speed.
4. **Lock exposure/focus/white balance** if the camera allows it; pumping
   exposure makes frames unusable between angles.
5. **Clear the shot** of hands, stands, straps and price tags.
6. **Capture the front square-on at least once** and note its frame index.
7. **Vary conditions on purpose** — lighting and background are variables, not
   nuisances to normalise away.

Accepted containers: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`.

## Annotating a clip

1. Measure the physical frame with calipers or from the manufacturer's spec
   **before** running any pipeline code, then write those numbers down.
2. Record only what you measured; omit the rest.
3. Set `front_frame` to the zero-based index of the square-on front frame.
4. Fill in `category`, `lighting`, `background` so results can be sliced.
5. Save as `<video-name>.json` beside the clip. `*.schema.json` files are always
   ignored by the loader.
6. Never edit an annotation to make a run look better. If a measurement is
   wrong, correct it and re-run the whole set; a hand-tuned annotation destroys
   the benchmark's only source of truth.

## Running the harness

```powershell
# 1. The honest empty case -- exits 3, reports "not run", never a pass.
venv\Scripts\python.exe scripts\benchmark_orbit_videos.py dataset\orbit_benchmark
#    -> status NOT_RUN, exit code 3, report.json + SUMMARY.md still written

# 2. The full comparison at 5, 8 and 10 views with explicit thresholds.
venv\Scripts\python.exe scripts\benchmark_orbit_videos.py dataset\orbit_benchmark `
    --output-dir output\benchmark\orbit `
    --views 5 8 10 `
    --target-fps 6.0 `
    --max-mae-mm 2.0 `
    --max-rmse-mm 3.0 `
    --max-error-mm 5.0 `
    --max-template-mismatches 0 `
    --min-confidence-level medium `
    --min-selected-views 5 `
    --min-view-class-diversity 3 `
    --min-acceptance-pass-rate 0.9

# 3. Skip the single-image baseline (recorded as "skipped", never faked).
venv\Scripts\python.exe scripts\benchmark_orbit_videos.py dataset\orbit_benchmark --no-front-image-baseline
```

`--reference-width-mm` re-expresses every dimension against a known real frame
width. Without a known reference the pipeline's own documented frame-width
assumption is what the millimetres mean, so an early run over a small set should
be treated as an internal consistency check, not as calibration.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Pass — at least one clip was scored **and** every enabled threshold check was evaluated and met |
| `2` | Fail — a scored clip or an evaluated threshold check failed, or a run errored |
| `3` | **Not run** — nothing was compared (empty or absent dataset, or no clip produced a usable measurement) |
| `4` | Usage error |

Exit code `3` is the important one: "we compared nothing" and "everything was
correct" must never look alike. Because of it, a CI job that expects `0` cannot
silently pass on an empty dataset.

A threshold check with no data is reported as **`not_evaluated`**, not as a pass.
`verdict.checks_evaluated` lists which checks actually had evidence, so a
five-check pass is distinguishable from a two-check pass. `verdict.accuracy_claimed`
is `true` only for a fully evaluated pass.

## The 5-vs-8-vs-10 comparison

The pipeline's production contract is 5–10 selected views, preferring 8. The
harness runs each clip at all three requested view counts — `target_views = 5, 8,
10`, with `min_views = 5` and `max_views = 10` left at the pipeline's
`FrameSelector` defaults — and reports each
separately rather than averaging them, because the interesting outcomes are
non-monotonic:

* **5 views** — the cheapest run. It answers "does the pipeline still work at the
  floor of the contract?" Fewer views means less evidence, so the fuser's
  confidence is capped and outlying views are harder to screen.
* **8 views** — the preferred configuration; this is the number the confidence
  scorer treats as full evidence.
* **10 views** — the ceiling. More views can add evidence, but also add
  low-quality frames and more chances for the selector to include a marginal
  pose, and cost grows roughly linearly with segmented frames.

The comparison is only meaningful once every clip has been run at all three
counts, so the report shows the per-count aggregates side by side and the
per-video rows underneath. A run that is missing a view count is visible as a
missing row, not as an average over whatever happened to succeed.

## Target dataset

**30 real annotated clips**, spanning:

| Axis | Required coverage |
| --- | --- |
| Material | metal, acetate, rimless |
| Rim thickness | thick, thin |
| Shape | rectangular, round, cat-eye, wayfarer, aviator |
| Conditions | varied lighting (diffuse, hard key, backlit, mixed daylight) and varied backgrounds (plain sweep, wood, cluttered retail shelf) |

Roughly three clips per axis point is the intent, so no single product, lighting
setup or frame family dominates the aggregate. A dataset that covers only metal
frames on a white sweep supports a claim about metal frames on a white sweep and
nothing else — slice the report by `category`, `lighting` and `background` before
quoting any number.

## How to read the report honestly

* `status = "not_run"` means nothing was compared. It is not a pass.
* `n/a` in a metric means there was no data, not a perfect score.
* `count = 0` in a per-dimension row means no annotation supplied that dimension.
* `template.accuracy` is computed only over clips that declare a `category`; a
  clip without one is excluded, never counted as a match.
* The confidence level is the pipeline's own self-assessment. It is reported
  next to the error, never instead of it: a "high" confidence with a large MAE is
  a pipeline bug, and the report is laid out so that is visible.
* A `PASS` acceptance status means the exported GLB satisfied the release
  contract for the mesh. It is not a statement about measurement accuracy.
* Errors over 30 clips say something about those 30 clips. They do not
  generalise to unseen products, unseen frame families or unseen capture
  conditions.

## Related files

* `scripts/benchmark_orbit_videos.py` — the harness
* `dataset/orbit_benchmark/README.md` — dataset layout and per-field schema reference
* `dataset/orbit_benchmark/annotations/EXAMPLE.schema.json` — annotation template (no real data)
* `tests/test_orbit_benchmark.py` — tests for the loader, the metrics and the empty-dataset verdict
* `docs/ORBIT_VIDEO_PIPELINE.md` — the pipeline this harness scores
* `scripts/benchmark_real_products.py` — the still-image equivalent, for a different capture mode
