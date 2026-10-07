# Orbit benchmark dataset

**This directory currently contains no real orbit videos and no annotations.**
There are zero annotated clips in the repository, so the benchmark has never been
run and **no accuracy claim is supported**.

That is the entire contents of this README's first section on purpose: the
quickest way to misuse this dataset is to point a harness at an empty folder and
read the resulting "0 errors" as a pass. `scripts/benchmark_orbit_videos.py`
refuses to do that — an empty dataset exits `3` with `status = "not_run"`.

## Layout

```
dataset/orbit_benchmark/
  README.md                      ← this file
  annotations/
    EXAMPLE.schema.json          ← TEMPLATE ONLY; contains no measurements
  <name>.mp4 + <name>.json       ← you add these (video + its annotation)
```

The harness pairs a clip with its annotation **by file stem**: `round_metal_01.mp4`
needs `round_metal_01.json`. Both files live in the dataset directory you pass on
the command line. `annotations/` holds templates and schema documentation only;
the harness does not read the dataset from it.

Accepted video containers: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`.

Anything unpaired is reported (`videos_without_annotation`,
`annotations_without_video`) rather than ignored, because a forgotten annotation
is the most common reason a real clip silently never gets scored.

## Annotation schema

Only **`front_frame`** and **`measurements`** are required for a scored
comparison. Everything else is optional and is recorded for slicing and review.

```json
{
  "product_id": "round_metal_01",
  "frame_count": 612,
  "fps": 30.0,
  "front_frame": 0,
  "side_frame": 153,
  "top_frame": 306,
  "usable_frames": [0, 5, 11, 18, 24, 153, 160, 306, 312],
  "expected_selected_views": 8,
  "category": "metal",
  "lighting": "diffuse indoor",
  "background": "plain white sweep",
  "measurements": {
    "frame_width": 138.4,
    "lens_width": 49.8,
    "lens_height": 42.1,
    "bridge_width": 20.6,
    "temple_length": 142.0,
    "rim_thickness": 1.1
  }
}
```

> The object above is a **shape illustration**. Its numbers are not a
> measurement of anything and must not be copied or aggregated. The only
> committed file that looks like an annotation is
> `annotations/EXAMPLE.schema.json`, which is an explicit template with null
> placeholders and no real data at all.

### Required vs optional

| Field | Required? | Type | Why |
| --- | --- | --- | --- |
| `front_frame` | **Yes** | int ≥ 0 | Zero-based index of the square-on front frame. The harness decodes it and runs the existing still-image pipeline on it to build the single-front-image baseline. |
| `measurements` | **Yes** (at least one dimension) | object of mm numbers | The only ground truth that is scored. |
| `measurements.<dimension>` | Per-dimension optional | number > 0, or `null`/absent | Omit what you did not measure. An omitted dimension is **excluded** from MAE/RMSE/max error; it is never scored as zero error. |
| `product_id` | Optional | string | Reporting and traceability. |
| `frame_count` | Optional | int | Sanity check against the decoded clip. |
| `fps` | Optional | number | Capture rate. |
| `side_frame` / `top_frame` | Optional | int / null | A clean profile and plan frame, for review. The video pipeline picks its own views; these do not constrain it. |
| `usable_frames` | Optional | int[] | Frames a human considers usable, for comparing against the pipeline's own quality gate. |
| `expected_selected_views` | Optional | int | Advisory expectation (normally 5–10). Recorded, not enforced. |
| `category` | Optional | enum | **The only field that makes template accuracy scorable.** Compared against the template the pipeline selected. A clip with no category is excluded from template accuracy instead of being counted as a match. |
| `lighting` | Optional | string | Slice the report by capture condition. |
| `background` | Optional | string | Slice the report by capture condition. |

`category` must be one of: `metal`, `acetate`, `rimless`, `thick`, `thin`,
`rectangular`, `round`, `cat_eye`, `wayfarer`, `aviator`.

Six dimensions are scored: `frame_width`, `lens_width`, `lens_height`,
`bridge_width`, `temple_length`, `rim_thickness`.

### Validation rules the loader enforces

* The file must exist and be valid JSON describing a JSON object.
* `front_frame` and `measurements` must be present.
* `front_frame` must be a non-negative integer (not a float, not a string).
* Every supplied measurement must be a finite number greater than zero.
  `null` and absent both mean "not supplied".
* At least one measurement must be usable; an annotation of six nulls is
  rejected, because it would otherwise score as an empty comparison.
* `category`, when present, must be a non-empty string. `frame_count`,
  `fps` and `expected_selected_views`, when present, must be numbers;
  `usable_frames`, when present, must be a list of integers.

## Recording a usable clip

Real footage only. Synthetic, rendered or AI-generated clips are **not**
acceptable input, because the benchmark's whole purpose is to predict behaviour
on real products.

1. **Mount the frame.** Put the eyewear on a stand or turntable at roughly eye
   height, with the frame centred and fully inside the picture at every angle.
   Never let a temple tip leave the frame edge; a cropped subject is rejected by
   the pipeline's visibility gate.
2. **Orbit slowly and steadily.** One full 360° revolution in 15–25 seconds. The
   selector prefers a spread of angles; a fast whip pan produces motion blur and
   near-duplicate frames.
3. **Lock exposure and focus if you can.** Auto-exposure pumping changes the
   silhouette between frames and the quality gate will drop the darker ones.
4. **Keep hands, straps, tags and props out of the shot.** Occlusion is the most
   common reason a clip scores nothing.
5. **Vary lighting and background across the set, deliberately.** The 30-clip
   target is not 30 clips of the same white sweep.
6. **Film the front view square-on at least once**, and note its frame index.
   That index is `front_frame`, and it is what the baseline is measured from.

## Annotating a clip

1. **Measure the physical frame first**, with calipers or from the manufacturer's
   specification, before looking at any pipeline output. Measure, then annotate —
   never the other way round.
2. Record **only** the dimensions you actually measured. Leave the rest out.
3. Find `front_frame` by stepping through the clip to the most square-on front
   view and note its zero-based index.
4. Fill in `category`, `lighting` and `background` so the report can be sliced.
5. Save as `<video-name>.json` next to the clip.
6. Note that the harness also ignores any `*.schema.json` file, so the committed
   template can never be picked up as a dataset annotation.
7. Validate:
   `venv\Scripts\python.exe -c "from pathlib import Path; from scripts.benchmark_orbit_videos import load_annotation; print(load_annotation(Path('dataset/orbit_benchmark/<name>.json')))"`

## Running the benchmark

```powershell
# Empty or absent dataset: exits 3 with status "not_run" -- deliberately not a pass.
venv\Scripts\python.exe scripts\benchmark_orbit_videos.py dataset\orbit_benchmark

# With real clips present:
venv\Scripts\python.exe scripts\benchmark_orbit_videos.py dataset\orbit_benchmark `
    --output-dir output\benchmark\orbit `
    --views 5 8 10 `
    --max-mae-mm 2.0 --max-rmse-mm 3.0 --max-error-mm 5.0 `
    --min-confidence-level medium --min-selected-views 5 `
    --min-acceptance-pass-rate 0.9
```

Outputs: `report.json` (machine-readable) and `SUMMARY.md` (markdown) in the
output directory, plus a console summary table.

Exit codes: `0` pass · `2` fail · `3` not run (nothing compared) · `4` usage error.

## Target dataset

30 real annotated clips, spanning:

| Axis | Coverage |
| --- | --- |
| Material | metal, acetate, rimless |
| Rim thickness | thick, thin |
| Shape | rectangular, round, cat-eye, wayfarer, aviator |
| Capture | varied lighting and varied backgrounds |

See `docs/ORBIT_VIDEO_BENCHMARK.md` for the full method, the 5-vs-8-vs-10
comparison and the current status statement.
