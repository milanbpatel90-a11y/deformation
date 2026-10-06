# Production readiness

The project uses product GLB templates and a parametric deformation engine. The
runtime path is image/measurement extraction, template matching, deformation,
material assignment, independent serialized-GLB checks, and browser VTO. GT_001
is the required default bundle; the exporter remains unchanged.

## Implemented gates

- GT_001 calibrates the supplied basis with the measured dimension Jacobian and
  rejects unsupported or coupled-infeasible dimensions.
- Its monotone regional field preserves a positive Jacobian. Runtime checks
  reject non-finite coordinates, degenerate triangle area, and edge stretch
  above the template's declared 3x limit. The bridge height transition is
  widened to avoid excessive local shear at the high supported settings.
- Every exported GLB is independently checked after serialization for glTF
  2.0, embedded buffer/accessor integrity, metres, normals, texture UVs when a
  material uses textures, valid indices, non-degenerate triangles, reasonable
  bounds, and requested frame/lens/bridge dimensions. A failed GLB and its
  sidecars are deleted.
- Each result has a deterministic `.manifest.json` with measurement units,
  source bundle hashes, output hash, deformation quality, and acceptance state.
- `DEFIRM_PRODUCTION_MODE=1` writes under `output/production` and requires exact
  Open3D per-component triangle self-intersection checks; failures do not leave
  a downloadable GLB. Default development outputs go under
  `output/development` and are marked `REVIEW` while exact intersections are
  not checked.
- GT_001 and six additional runtime bundles have a committed asset checksum
  manifest. The required Gold GLB and basis are now included through Git LFS.
- The real-product benchmark runner reports template accuracy, measurement MAE,
  RMSE, maximum error, failures, and per-model output acceptance.
- Orbit video (`POST /api/deform/video`) decodes a clip, gates frames on blur,
  glare and exposure, selects 18-24 well-spread views, classifies them, measures
  each view, and fuses with a weighted median plus MAD screening that rejects
  outlier views. It reports the per-dimension median, spread, agreement and every
  rejected value. See `docs/ORBIT_VIDEO_PIPELINE.md`; note that the automatic
  scale caveat above applies unchanged, and that front/rear separation is not
  claimed because the one-class mask is a solid silhouette.

## Release gate

Run `python -m scripts.check_readiness`. The `ready` field means the service has
the required local assets and passes its GT_001 dimensional smoke checks.
`production_ready` is a stricter release decision. It remains false until all
of the following evidence is present:

1. Start the service with `DEFIRM_PRODUCTION_MODE=1` and use that profile for
   release generation. This requires Open3D; unsupported/missing exact checks
   fail closed.
2. Run `python -m scripts.benchmark_real_products` with at least 100 real,
   independently measured products and product-owner-approved thresholds.
   Keep customer images outside Git unless their owner approved repository
   storage. The benchmark must show no template mismatches or review-only GLBs.
3. Record browser and mobile VTO acceptance in
   `dataset/real_products/release_acceptance.json` after testing real devices.
4. Review lens/rim clearance, temple-joint fit, material segmentation, and all
   collision findings on the intended template families.

These release requirements are intentional: automatic measurements from photos
without a scale reference are estimates, and passing synthetic geometry tests
does not establish real-product accuracy. A rectangular template also preserves
its authored silhouette; it cannot become an unrelated aviator, round, or
cat-eye product from dimension changes alone.

## Run the local checks

```powershell
python -m pytest -q --tb=short
python -m scripts.template_asset_manifest
python -m scripts.check_readiness
python -m scripts.validate_production_glb
python -m scripts.benchmark_rim_deformer --vertices 61872
```

Install/fetch Git LFS assets after cloning (`git lfs install`, `git lfs pull`).
The asset manifest detects missing, incorrect, or incomplete LFS downloads.

## Deployment controls

Uploads are limited to 10 MiB and 16 megapixels per image, with at most six
images. CPU deformation jobs are serialized per process and concurrent requests
receive 503 with `Retry-After`. Before public deployment, configure a reverse
proxy request-size limit, timeouts, authentication, rate limits, and output
retention/disk monitoring. Output URLs are bearer links, not per-user access
control. The viewer uses external Three.js and MediaPipe CDNs; production
browser/mobile, webcam, load, and GPU memory behavior still require device QA.
