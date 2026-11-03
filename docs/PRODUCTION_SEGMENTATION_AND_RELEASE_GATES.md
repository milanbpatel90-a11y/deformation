# Production segmentation and release gates

## Runtime configuration

For a production API, configure both variables:

- `DEFIRM_YOLO_MODEL=/absolute/path/to/glasses_seg.pt`
- `DEFIRM_REQUIRE_YOLO=1`

The first points to the fine-tuned eyewear checkpoint. The second makes application startup fail if no valid checkpoint can be loaded. This is intentional: a public production endpoint must not silently generate customer GLBs using the low-confidence OpenCV threshold fallback.

For local development only, leave `DEFIRM_REQUIRE_YOLO` unset to retain the fallback. The health endpoint should still report a degraded state when the fallback is active. Do not treat a degraded service as production-ready.

## Supported model contracts

The segmentation loader accepts:

1. **One-class silhouette model** — one class representing the complete eyewear silhouette. This can support overall outline and coarse measurements, but cannot independently segment rims, bridge, left/right lenses, temples, or nose pads.
2. **Six-class part model** — exact class IDs and labels:
   - 0: `eyewear_rim`
   - 1: `eyewear_temple`
   - 2: `bridge`
   - 3: `left_lens`
   - 4: `right_lens`
   - 5: `nose_pad`

Do not use a one-class silhouette checkpoint as if it were a six-part model. Do not silently remap class IDs. If the dataset/model uses another order, update and validate the contract deliberately.

## Dataset QA before training

The real-eyewear dataset importer/validator checks file presence, image-mask dimensions, class IDs, and polygon syntax. Those checks are structural only. AI-generated prelabels are not ground truth.

Before a model is used for production:
- Human-review every training and validation mask, prioritizing thin rims, transparent lenses, metal frames, nose pads, hinges, dark-on-dark products, and complex backgrounds.
- Keep product identities separated between train, validation, and test to avoid near-duplicate leakage.
- Keep the test split out of training and hyperparameter tuning.
- Evaluate class-wise precision, recall, mask mAP50 and mAP50-95 on human-verified labels.
- Inspect overlays and false positives/negatives; aggregate metrics alone do not prove thin-part quality.
- Save the dataset revision, annotation reviewer status, training config, checkpoint hash, and evaluation report with each model release.

A single filled polygon for the whole eyewear object cannot accurately represent interior holes or distinguish the frame from transparent lens openings. Use separate part masks / a suitable multi-class or instance-segmentation annotation format when the downstream deformation needs those parts.

## Physical measurement calibration

Image segmentation gives pixel geometry, not millimetres by itself. Absolute measurements require a known reference in the same plane or reliable manufacturer specifications. For image uploads, supply the known full frame width when available. Otherwise the result must remain marked as assumed/uncalibrated and must not be advertised as dimensionally accurate.

Perspective, lens tilt, camera distortion, and different object depths can make a single pixel-to-mm scale invalid across all views. Prefer calibrated front, side, and top measurements, then fuse compatible dimensions with outlier checks.

## Release acceptance suite

Run the following on a held-out set of at least 100 distinct real eyewear products, with 4–5 views and manufacturer dimensions wherever possible:

- Segmentation overlays visually reviewed and recorded.
- Frame width, lens width/height, bridge width, temple length, and rim thickness compared with reference measurements; report MAE, median absolute error, and 95th percentile error per dimension.
- Confirm invalid/ambiguous inputs fail closed with actionable errors.
- Confirm the selected template has distinct required parts and compatible topology.
- Compare pre-deformation and post-deformation vertex coordinates and per-part bounds to prove the intended geometry actually changed.
- Reload each exported GLB and validate units (metres in glTF), finite vertices, non-empty indices, normals, UVs/materials where required, and expected part names.
- Open generated GLBs in the actual browser viewer and verify scale, face fit, orientation, materials, and occlusion on multiple face shapes.
- Record pass/fail by product; do not report only the overall success percentage.

A green unit-test suite is necessary but is not a substitute for this real-product acceptance test.
