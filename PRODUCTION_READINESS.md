# Deployment status

**Gold Template GT_001 is installed and available for generation.**

The complete bundle metadata and coupled basis were recovered from the local
`gt001_delivery` package. The original Gold Template runtime GLB was installed at
`assets/templates/GT_001/geometry/template.glb`; its vertices and face topology
match the basis exactly. The API discovers asset bundles as well as legacy templates.
Gold Template uses its supplied basis to deform combined frame/rim/bridge regions,
so it does not require those regions to be separate GLB meshes.

`python -m scripts.check_readiness` and `/readyz` now report GT_001 as ready.
Readiness validates structural compatibility, not full-range visual fidelity.
The five primary manual dimensions drive the basis; rim thickness and temple
curve retain the supplied template geometry, with this limitation returned in
quality warnings. Manual dimensions outside the basis's supported ranges are
rejected rather than silently clamped. Large deformations still require visual QA.

The older `templates/geometric_metal.glb` and `rectangle_plastic.glb` entries remain
unavailable through the separate-part engine. They are distinct from GT_001.
`assets/.gitignore` intentionally excludes large binaries: deployment must copy
`geometry/template.glb` and `deformation/basis.npz` along with the JSON metadata,
or store those binaries through the project's asset/LFS delivery process.

## Manual measurement contract

The viewer suggests editable frame width, lens width, lens height, bridge width,
temple length and rim thickness after image upload. These are estimates using an assumed scale, not exact physical measurements. Manual editing is supported and automatic template selection checks supported ranges. POST /api/measurements/suggest accepts images and returns suggested values, range limits and any adjustments. Photos are optional. Generation image endpoints require a
`measurements` form field containing a Measurements JSON object; the five primary
dimensions are mandatory. These values bypass image-based dimensional estimation
and multi-view fusion. Images can still provide contours and lens appearance.
The measurements-only endpoint accepts form fields. All dimensions must be finite
and positive; invalid inputs return 422. Legacy Python callers can still explicitly
use the experimental estimation path by omitting `manual_measurements`.

## Run and verify

Use a dedicated Python 3.11 environment and install `requirements-dev.txt`.
This workspace's populated environment is `venv`; `.venv` was missing runtime packages.

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m scripts.check_readiness
.\venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Serve the viewer at `/viewer/` from the same origin. For a separate trusted frontend,
set `DEFIRM_CORS_ORIGINS` to a comma-separated origin list. Set `DEFIRM_YOLO_MODEL`
to a validated eyewear segmentation model. The generic COCO model is no longer
preferred over the trained eyewear model.

Uploads are limited to 10 MiB and 16 megapixels per image, with at most six images.
Heavy inference/deformation jobs run outside the event loop, one at a time per
process; concurrent jobs receive 503 with Retry-After. Configure a reverse proxy
with an overall request-body limit (65 MiB), timeouts, authentication and rate
limits before public deployment. Downloads are bearer URLs, not per-user access
control. Define output retention and disk monitoring for the deployment.

The viewer still depends on external Three.js and MediaPipe CDNs. Webcam operation
requires HTTPS or localhost and browser permission. Camera/browser visual QA,
load testing, artist-asset fidelity, and Blender-only factory/add-on execution
remain deployment checks. Python compilation does not exercise Blender APIs.

Component tests use a temporary procedural scene with separate parts; regression
tests independently verify rejection of ambiguous geometry, manual input handling,
API errors and limits, GLB materials and export. One existing Gold Template
integration test remains skipped pending asset preparation. Passing those tests
does not replace production visual and load testing.
