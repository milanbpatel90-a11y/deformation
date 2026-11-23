# Video-to-Reference Extraction — Deliverables Report

Milan Patel

## 1. Branch

Branch: `feature/video-to-reference-extraction`, pushed to `origin`. **`main` is untouched** — its remote tip was never advanced by this work (confirmed by `git ls-remote` and `git diff` against the common ancestor commit).

## 2. Files created

**`backend/video_extraction/`** (new package, ~2,400 lines):

| File | Responsibility |
| --- | --- |
| `config.py` | All thresholds/limits as `VIDEOX_*` env vars |
| `video_processor.py` | Upload validation (extension, magic bytes, size, codec/resolution/fps/duration via ffprobe), streamed frame sampling |
| `frame_extractor.py` | Adaptive motion-aware sampling (not every-Nth-frame) |
| `quality_analyzer.py` | Laplacian sharpness, exposure, mask-localized glare |
| `glasses_detector.py` | Glasses presence/bbox coverage check ahead of full segmentation |
| `view_classifier.py` | front / front_left_45 / front_right_45 / side_left / side_right / top / unknown, with confidence |
| `duplicate_filter.py` | Perceptual-hash near-duplicate rejection |
| `frame_ranker.py` | View-diversity-aware best-frame selection |
| `segmentation_processor.py` | Reuses the existing YOLO `best.pt` / `GlassesSegmenter`, 6-class mapping (rim, temple, bridge, left_lens, right_lens, nose_pad) |
| `measurement_preprocessor.py` | Builds pixel/estimated/calibrated measurement bundle for the deformation engine |
| `extraction_pipeline.py` | Orchestrates the whole job: sample → filter → classify → segment → rank → report |
| `jobs.py` | Bounded thread-pool background job runner (keeps FastAPI's event loop unblocked) |
| `db.py` | Per-job state (SQLite), merchant-scoped |
| `storage.py` | Private, per-merchant, path-traversal-safe file storage (local disk; see §7) |
| `schemas.py` | Pydantic models (`ViewLabel`, `CalibrationStatus`, `JobStatus`, measurement/report schemas, etc.) |
| `exceptions.py` | Typed error hierarchy (no bare exceptions reaching the API layer) |
| `README.md` | Feature docs (recording guidance, config, workflow, deployment, limitations) |

**`backend/api/video_extraction_routes.py`** — the authenticated FastAPI router (§5).

**`tests/video_extraction/`** — `conftest.py`, `test_api.py`, `test_duplicate_and_ranking.py`, `test_storage_and_db.py`, `test_video_processor.py` (38 tests).

**`docs/VIDEO_EXTRACTION.md`** — merged in from a parallel branch attempt (§12): recording guidance and deployment notes.

## 3. Files modified

| File | Change |
| --- | --- |
| `backend/api/main.py` | Registers the new `video_extraction_routes` router; adds it to the `/api` info listing |
| `backend/pipeline/deformation_pipeline.py` | Additive-only: a `return_quality` flag on `run_from_multiple_images` (rejects failed geometry before GLB export instead of exporting unconditionally) and a `strict_validation` flag on `run_from_measurements` (refuses the silent uniform-scale fallback). Both default to `False`; both existing `/api/deform*` callers are byte-for-byte unchanged |
| `backend/segmentation/segmenter.py` | Additive `segment_detailed()` returning per-class masks/confidences from the existing `best.pt`; the original `segment()` used by `/api/deform*` is untouched |
| `requirements.txt` | Added `ImageHash>=4.3.2` only — no COLMAP, no Nerfstudio, no paid APIs |
| `viewer/index.html` | New "Upload Video" dashboard section (§6) |

No existing production templates, descriptor logic, or YOLO model were deleted or replaced.

## 4. Architecture and processing flow

1. **Upload** — merchant uploads MP4/MOV; extension, magic bytes, codec, size, resolution, fps and duration are validated before anything is trusted; file is streamed to private, merchant-scoped storage (never loaded whole into memory).
2. **Process (background job, thread pool)** — adaptive frame sampling detects meaningful orientation change rather than grabbing every Nth frame.
3. **Per-candidate filtering** — glasses-presence/bbox-coverage check, Laplacian sharpness, exposure, glare, perceptual-hash duplicate rejection — all thresholds configurable, none hardcoded to one resolution.
4. **View classification** — front / front_left_45 / front_right_45 / side_left / side_right / top / unknown, each with a confidence score; low-confidence frames fall to `unknown` rather than guessing, and stay eligible for manual merchant approval.
5. **Ranking & selection** — diversity-aware: prioritizes covering distinct views over hoarding near-identical fronts.
6. **Segmentation** — reuses the existing YOLO `best.pt` and its 6 real classes; a missing component is recorded as missing/low-confidence, never fabricated.
7. **Measurement preprocessing** — pixel vs. estimated vs. calibrated dimensions are explicitly distinguished; nothing uncalibrated is ever labeled as millimetres.
8. **Package & reports** — writes `extracted/`, `masks/`, `overlays/`, `measurements.json`, `extraction_report.json`, `quality_report.json` per job; only selected views are saved, missing views and warnings are reported, not hidden.
9. **Merchant review & approval** — dashboard gallery; nothing is auto-approved.
10. **Hand-off to the existing deformation engine** — approved references go into the existing template-selection + `MeshDeformer` path via `run_from_multiple_images(..., return_quality=True)`; a failed quality check raises instead of exporting, so invalid geometry never reaches a GLB file.

## 5. API endpoints

All under `/api/video-extraction`, all authenticated, all ownership-checked (`_authorize_job`, 404 — not 403 — on cross-merchant access to avoid leaking existence), all errors returned as structured JSON with tracebacks stripped and no filesystem paths exposed.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/upload` | Upload an MP4/MOV, creates a job |
| POST | `/{job_id}/process` | Starts background extraction (optional known-frame-width calibration) |
| GET | `/{job_id}/status` | Poll job state/progress |
| GET | `/{job_id}/results` | Extraction report + measurements once ready |
| GET | `/{job_id}/frames` | Candidate/selected frame metadata |
| GET | `/{job_id}/files/{relative_path}` | Serves one file from the job's private package (ownership-checked, path-traversal-safe) |
| POST | `/{job_id}/approve` | Merchant-approved frames → existing deformation pipeline → validated GLB |
| POST | `/{job_id}/regenerate` | Re-run extraction |
| DELETE | `/{job_id}` | Delete a job and its files |
| GET | `` (root) | List the caller's own jobs |

## 6. Dashboard changes

Added an "Upload Video" section to `viewer/index.html` (hidden for the read-only `viewer` role, same as other generation controls):

- Drag-and-drop or click-to-choose MP4/MOV upload with a progress bar
- Optional known-frame-width (mm) calibration input before processing
- Live status polling while the job processes
- Frame gallery with per-frame view label and quality score, overlay thumbnail
- Missing-view warnings surfaced from the extraction report
- Click-to-approve frame selection (nothing is pre-selected — the merchant must actively choose frames; "Approve" stays disabled until at least one is picked)
- "Approve Selected & Generate 3D Model" → calls `/approve` and loads the resulting GLB straight into the existing viewer
- "Re-run extraction" (regenerate)

**Known gap:** there is no dedicated "swap this frame for a different candidate" control — only approve/reject of the ranked candidates. Noted as a limitation in §9 rather than left silent.

## 7. Storage changes

The repository had **no existing Wasabi/S3 integration** to reuse (confirmed by grep across the codebase before building — this is stated plainly in the code/docs rather than silently assumed). `backend/video_extraction/storage.py` is a private, local-disk implementation behind a swappable interface: per-merchant object keys, 0700/0600 permissions, served only through the authenticated + ownership-checked `/files/{relative_path}` route (never a public path), with reliable temp-file cleanup. `docs/VIDEO_EXTRACTION.md` documents the env vars (`VIDEO_STORAGE_BACKEND=s3`, `WASABI_BUCKET`, `WASABI_ENDPOINT_URL`, `WASABI_REGION`, `WASABI_ACCESS_KEY_ID`, `WASABI_SECRET_ACCESS_KEY`) a real Wasabi-backed implementation would need, since Wasabi itself isn't wired into this repo yet.

## 8. Actual test execution results

Both runs executed in this session, not claimed from memory:

```
cd deformation && python3 -m pytest tests/video_extraction -q
......................................   [100%]
38 passed in 54.61s
```

```
cd deformation && python3 -m pytest -q   # full repo suite, regression check
..........................................s..........................  [87%]
..........                                                              [100%]
81 passed, 1 skipped in 16.46s
```

`tests/video_extraction/test_api.py` specifically covers: upload validation (wrong extension, fake-content-with-real-extension, valid tiny video), cross-merchant IDOR (404 on another merchant's job for both read and delete, admin-can-read-any), job-id format validation, the full status state machine, and the approve flow including both a quality-gate-failure case and a success case. The other three files cover frame extraction/duplicate/ranking behavior and storage/DB path-traversal and persistence. The full-suite run confirms the existing `/api/deform*` GLB-generation path still passes unchanged.

## 9. Known limitations

- **No reference-marker/known-object calibration** — only a manually entered known frame width (mm). A physical calibration marker in-frame is not yet auto-detected.
- **Storage is local-disk**, not actually backed by Wasabi (§7) — the repo had nothing to reuse yet.
- **View classification is a documented heuristic** (bbox aspect ratio + left/right mask symmetry), not a pose model. It is intentionally conservative: anything it isn't confident about is labeled `unknown` with its confidence score, for manual merchant approval, rather than guessing.
- **No per-view frame swap in the dashboard** (§6) — merchants approve/reject ranked candidates, but can't yet pick a different timestamp for a given view from the raw video.
- **Rim thickness and projected temple extent** are explicitly flagged as low-confidence approximations in the extraction report, not presented as precise measurements.
- **Front-view calibration cannot calibrate side/top views** — each view's measurements carry their own calibration status rather than inheriting the front view's.

## 10. Deployment instructions

1. Install `ffmpeg`/`ffprobe` in the Railway image (needed for robust codec/metadata inspection).
2. `pip install -r requirements.txt` picks up the one new dependency, `ImageHash`.
3. Set the `VIDEOX_*` env vars in `backend/video_extraction/config.py` / `docs/VIDEO_EXTRACTION.md` to size limits appropriate for the deployment (max upload bytes, duration, resolution, fps, concurrent jobs).
4. Put the job database path (`VIDEO_EXTRACTION_DB_PATH`) on a persistent Railway volume — job state must survive restarts.
5. For real object storage, set `VIDEO_STORAGE_BACKEND=s3` plus `WASABI_BUCKET`, `WASABI_ENDPOINT_URL`, `WASABI_REGION`, `WASABI_ACCESS_KEY_ID`, `WASABI_SECRET_ACCESS_KEY` and keep the bucket private; otherwise it defaults to local disk, which is development-only.
6. No GPU or extra model download is required — the existing `models/best.pt` is reused as-is.
7. No new paid APIs, no COLMAP, no Nerfstudio — nothing new to provision there.

## 11. Example extraction report

Illustrative `extraction_report.json` shape, matching `schemas.ExtractionReport`:

```json
{
  "job_id": "a1b2c3d4-...",
  "video_metadata": {
    "duration_seconds": 18.4,
    "fps": 30,
    "resolution": [1920, 1080],
    "codec": "h264"
  },
  "candidates_sampled": 142,
  "candidates_rejected": 118,
  "selected_views": ["front", "front_left_45", "side_right", "top"],
  "missing_views": ["front_right_45", "side_left"],
  "warnings": [
    "front_right_45 not found with sufficient confidence; try slowing down that turn"
  ],
  "measurement_warnings": [
    "bridge_width is pixel-only (no calibration supplied); not passed to the deformation engine as millimetres"
  ],
  "calibration_status": "estimated"
}
```

Real jobs will also include a `quality_report.json` (per-frame sharpness/exposure/glare/confidence scores) and `measurements.json` (one `MeasurementValue` per dimension, each with `value`, `unit`, `confidence`, `source_view`, `calibration_status`).

## 12. Reconciling with a parallel branch attempt

The remote `feature/video-to-reference-extraction` branch had diverged with a separate, thinner attempt at the same feature (no API routes, smaller per-module implementations, a different function-naming scheme). Rather than force-overwriting either side's work, this was resolved as a merge: every `backend/video_extraction/*` add/add conflict was kept as this branch's complete, tested implementation; the remote's independently useful additions were folded in (an additive `segment_detailed()` on the existing segmenter, an additive `strict_validation` flag on `run_from_measurements`, and `docs/VIDEO_EXTRACTION.md`); its one test file was dropped because it exercised a different, incompatible API that doesn't exist here and would only fail. Full suite reverified after the merge (§8) before pushing.
