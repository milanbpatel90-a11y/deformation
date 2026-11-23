# Video-to-reference extraction

Lets a merchant upload a video of a physical pair of glasses instead of
individually-framed product photos. The backend extracts, filters, and
classifies candidate frames, lets the merchant review/approve the best
reference images per viewing angle, and then hands those approved images to
the existing, **unmodified** `DeformationPipeline.run_from_multiple_images`
to generate and validate a GLB — exactly the same code path the manual
multi-image upload (`POST /api/deform/multi-view`) already uses.

This package does **not** attempt full 3D reconstruction from video
(no COLMAP, no NeRF/Gaussian-splatting, no paid APIs). It is a smarter
frontend for the template-deformation pipeline that already exists.

## Architecture / processing flow

```
video_processor   -- validate upload (ext, magic bytes, size), open with
                      cv2.VideoCapture, validate duration/resolution/fps,
                      stream frames sequentially (grab/retrieve; never the
                      whole file in memory)
frame_extractor   -- adaptive, motion-aware sampling (not fixed every-Nth)
glasses_detector  -- reuses backend.segmentation.segmenter.GlassesSegmenter,
                      pointed at models/best.pt (the real 6-class trained
                      model), for per-component masks + confidences
quality_analyzer  -- resolution-normalized sharpness (Laplacian variance),
                      exposure histogram, mask-localized glare
view_classifier   -- heuristic view bucket from mask aspect ratio / L-R
                      symmetry (NOT a calibrated pose estimator)
duplicate_filter  -- perceptual hash (imagehash.phash) dedup within a view
frame_ranker      -- diversity-aware: best N per view bucket, not globally
                      top-scoring (so one dominant angle can't starve the
                      others the deformation pipeline actually needs)
segmentation_processor -- persists original/overlay/cutout/per-class-mask
                      images for every selected frame via storage.py
measurement_preprocessor -- reuses ShapeClassifier + MeasurementExtractor +
                      MeasurementFuser (unchanged) to fuse selected frames
                      into provenance-tagged measurements
extraction_pipeline -- orchestrates the above into one job run, writes
                      extraction_report.json / quality_report.json /
                      measurements.json
jobs.py           -- bounded ThreadPoolExecutor background runner (no
                      Celery/Redis in this repo; see "Known limitations")
db.py             -- SQLite job-state table (same pattern as backend/auth.py)
storage.py        -- private per-job local-filesystem tree; every file is
                      served only through the authenticated, ownership-
                      checked `/api/video-extraction/{job_id}/files/...`
                      route (see "Known limitations" re: Wasabi/S3)
```

On approval (`POST /{job_id}/approve`), the route calls
`DeformationPipeline.run_from_multiple_images(..., return_quality=True)`.
That flag (additive, defaults to `False` so the two pre-existing
`/api/deform*` endpoints are byte-for-byte unchanged) makes the pipeline
raise instead of exporting when the already-computed `QualityChecker`
result's `passed` is `False` — so a GLB is only ever written after its own
geometry quality validation succeeds, and the failure reason reaches the
merchant instead of a silently-exported bad mesh.

## API endpoints

All routes require authentication; the uploader's own account owns a job,
and only that account or an `owner` (admin) role may read/act on it (every
route 404s rather than 403s for someone else's job, so existence is never
leaked). Mutating routes require the `owner`/`editor`/`operator` roles
(same tier as the existing `/api/deform*` endpoints); `viewer` accounts get
read-only access to their own jobs.

| Method & path | Purpose |
| --- | --- |
| `POST /api/video-extraction/upload` | Validate + store an MP4/MOV. |
| `POST /api/video-extraction/{job_id}/process` | Start background extraction (optional `known_frame_width_mm` form field for calibration). |
| `GET /api/video-extraction/{job_id}/status` | Poll job status/progress. |
| `GET /api/video-extraction/{job_id}/results` | Report + measurements + selected frames (once past processing). |
| `GET /api/video-extraction/{job_id}/frames` | Just the selected frames. |
| `GET /api/video-extraction/{job_id}/files/{path}` | Serve one stored asset (image/mask/overlay/cutout/report JSON). |
| `POST /api/video-extraction/{job_id}/approve` | Merchant approves a set of `frame_index` values -> runs the real deformation pipeline -> GLB. |
| `POST /api/video-extraction/{job_id}/regenerate` | Re-run extraction from the stored video (e.g. after changing calibration). |
| `DELETE /api/video-extraction/{job_id}` | Delete a job and its files. |
| `GET /api/video-extraction` | List the caller's own jobs. |

## Configuration / environment variables

Every threshold has an env-var override; see `config.py` for the full,
documented list (prefix `VIDEOX_`), e.g.:

- `VIDEOX_STORAGE_ROOT` (default `storage/video_extraction`)
- `VIDEOX_DB_PATH` (default `data/video_extraction.sqlite3`)
- `VIDEOX_MAX_UPLOAD_MB` (default `300`)
- `VIDEOX_MIN_DURATION_SEC` / `VIDEOX_MAX_DURATION_SEC` (default `1.5` / `120`)
- `VIDEOX_MIN_RESOLUTION` / `VIDEOX_MAX_RESOLUTION` (default `180` / `4096` px)
- `VIDEOX_SHARPNESS_MIN` (default `45.0`, Laplacian-variance units at an 800px-wide reference frame)
- `VIDEOX_MAX_SELECTED_PER_VIEW` (default `2`)
- `VIDEOX_WORKER_POOL_SIZE` / `VIDEOX_MAX_JOBS_PER_USER` (default `2` / `3`)
- `DEFIRM_YOLO_MODEL` — explicit override for which segmentation weights to use (defaults to `models/best.pt` for this feature specifically; does not change what `/api/deform*` resolves to).

## Video recording guidelines (merchant-facing)

- MP4 or MOV, 1.5–120 seconds, 180px–4096px on a side, 4–120 fps.
- Slowly rotate the glasses (or walk the camera around them) covering
  front, both 45° angles, both sides, and top if possible — the ranker
  keeps the best frame(s) per angle, not just the sharpest overall.
- Even, diffuse lighting; avoid direct reflections off the lenses/metal.
- Keep the glasses large in frame but fully inside it (very small or
  cropped glasses are rejected by the area-fraction gate).
- For millimetre-accurate (not just estimated) measurements, supply the
  known total frame width in the calibration field.

## Local testing

```bash
pip install -r requirements.txt
pytest tests/video_extraction -q
```

The suite (`tests/video_extraction/`) isolates every test behind its own
temp storage root + SQLite path (`conftest.py`), stubs the background job
runner to execute synchronously, and stubs `DeformationPipeline` for the
approve-flow tests (the real 3D engine already has its own regression
suite under `tests/test_*`). It does exercise the real upload-validation
and real YOLO-segmentation code paths against a tiny synthetic video.

## Railway deployment

No new infrastructure is required: the feature runs in the same single
FastAPI process, using a bounded in-process thread pool instead of a
separate worker/queue service. Ensure the Railway volume (or equivalent
persistent disk) covers both `storage/video_extraction/` and
`data/video_extraction.sqlite3` — on an ephemeral filesystem, in-flight
and completed jobs are lost on redeploy (see "Known limitations").

## Known limitations

- **No Wasabi/S3 integration exists in this repository.** The task brief
  assumed one; `storage.py` is a thin, swappable abstraction (`save` /
  `open_path` / `delete` / `local_url`) backed by local disk today, with
  every file still served only through the authenticated/ownership-checked
  `/files/...` route. Swapping in a real object-storage backend later
  (signed URLs from `local_url`) requires touching only `storage.py`.
- **No Celery/Redis exists in this repository.** Background processing
  runs on an in-process `ThreadPoolExecutor` (`jobs.py`). Job *state*
  survives a process restart (SQLite), but in-flight work does not resume —
  a job that was "processing" when the process died must be re-submitted
  via `/regenerate`.
- **View classification is a heuristic**, not a calibrated pose estimator
  (mask aspect ratio + left/right symmetry). Low-confidence frames are
  labeled `unknown` and still shown to the merchant rather than guessed.
- **No reference-marker/calibration-object detector.** Only manual
  `known_frame_width_mm` entry is supported; without it, measurements are
  tagged `calibration_status=estimated`, never silently relabeled as exact
  millimetres.
- **No multi-tenant "merchant" table.** Ownership is keyed to this app's
  existing `editor`/`operator`/`viewer`/`owner` accounts (`owner_user_id`),
  not a separate merchant/store concept — there is none in this codebase.
