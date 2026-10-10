# Video pipeline consolidation record

This repository grew **four rival 360-degree video implementations** in parallel.
This document records what existed, which one became canonical, what was absorbed
from each, and what was deliberately left behind.

## What existed

| # | Location | State | Approach |
|---|---|---|---|
| A | `backend/video_pipeline.py` on `feature/video-to-3d-pipeline` (266 lines) | committed, unmerged | 36 `linspace`-sampled frames via random access, `view_score` ranking, top-8, plain median fusion, optional `reference_width_mm`, `.video.json` sidecar |
| B | `backend/video_pipeline.py` on `prod/production-hardening-v2` (344 lines) | committed, unmerged (PR #15) | A plus angular bins, `_select_angular_coverage`, a minimum occupied-sector check, `quality_score`, and new scale fields added to the core `Measurements` model |
| C | `backend/video/{extract_frames,quality_gate,angle_classifier}.py` on `production-hardening-2026-10` (91 lines) | committed, unmerged (PR #16) | sequential every-3rd-frame decode, Laplacian/glare gate, greedy temporal gap, default target 5, signed left/right yaw |
| D | `backend/video/*` + `backend/pipeline/video_pipeline.py` | this branch | calibrated gate, temporal-median descriptors, duplicate clustering, farthest-point coverage selection, weighted-median + MAD fusion with per-dimension view routing |

Path collisions were real, not hypothetical: `backend/video_pipeline.py` existed in
two incompatible versions, and `backend/video/quality_gate.py` in two more.

## Canonical choice

**D**, extended with the good ideas from A, B and C, is the single implementation.
It was chosen because it is the only one that already satisfied the hard parts of
the specification:

* a calibrated quality gate that *rejects* frames (A and B only ranked them),
* duplicate removal (absent from A and B; C had a temporal gap rather than
  similarity clustering),
* coverage-aware selection (B introduced angular bins; A ranked by front width,
  which clusters the selection wherever the lighting was best),
* robust fusion with MAD outlier screening and per-dimension view routing
  (A and B used a plain median across all views; C had no fusion at all),
* a bounded candidate pool so segmentation never runs on the whole clip.

There is now exactly **one** video entry point: `backend/pipeline/video_pipeline.py`
(`VideoDeformationPipeline`) over the `backend/video/` package. No
`backend/video_pipeline.py` module exists on this branch, so
`import backend.video_pipeline` cannot silently resolve to a second engine.

## Absorbed

* **From A/B** -- an optional caller-supplied `reference_width_mm`, with a
  plausibility range (20-250 mm). This is how absolute scale is exposed rather
  than implied.
* **From A/B** -- the `.video.json` sidecar next to the GLB, and the
  `capture_mode` / `reconstruction: false` framing. The sidecar is deliberately a
  *separate file* from the deformation pipeline's own `.manifest.json`, which is
  the release contract for the exported mesh and must not be rewritten by a
  capture-mode concern.
* **From B** -- the minimum occupied-angular-sector requirement, now expressed as
  sectors over the clip plus a temporal-span ratio, both reported in the manifest.
* **From C** -- the left/right perspective signal, implemented as a documented
  signed `lateral_skew` convention rather than a yaw estimate, and the preference
  for a ~5-view floor.

## Deliberately not carried forward

* **B's extension of the core `Measurements` model** with `measurement_scale_*`
  fields. Scale provenance is capture metadata, not a property of an eyewear
  measurement, and adding fields to the model that the deformers consume risks
  touching the engine this task must not disturb. Scale lives in the video
  manifest instead.
* **C's Celery/Redis asynchronous job API** and the surrounding Docker/CI/dashboard
  scaffolding from PR #16. Out of scope here and unmerged; this branch keeps the
  existing synchronous, serialized-job API contract.
* **C's gate thresholds** (`min_blur=40`, `max_glare=0.08`). Superseded by
  thresholds calibrated against `test_images/*.jpg` and synthetic degradations,
  with the measured accept/reject table recorded in
  `docs/ORBIT_VIDEO_PIPELINE.md`.
* **A and B's random-access sampling** (`cap.set(CAP_PROP_POS_FRAMES)`) in favour
  of sequential `grab`/`retrieve` sampling on a time grid, which does not assume a
  constant frame rate and never seeks backwards.

## Merge note

If PR #15 or PR #16 is merged later, expect conflicts at
`backend/video_pipeline.py` (delete it: superseded) and
`backend/video/quality_gate.py` (keep this one, it is the calibrated version).
`tests/test_video_pipeline.py` is also claimed by A; this branch's version is the
comprehensive one (66 tests).
