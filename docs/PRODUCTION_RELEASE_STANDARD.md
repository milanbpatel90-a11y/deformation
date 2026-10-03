# Production Release Standard

This repository uses a fail-closed release contract for generated eyewear GLBs.

## What can be a production PASS

A generation may be released only when all of these are true:

1. The selected template is a verified basis/production bundle with real binary assets.
2. The deformation engine accepts the requested dimensions within the template's coupled constraints.
3. The deformation has finite coordinates, non-degenerate triangles, positive continuous Jacobian, and no edge-stretch violation.
4. The serialized GLB is independently validated as glTF 2.0 with metres as world units, millimetres as measurement units, valid indices, normals, and required UV data when textures are used.
5. Serialized dimensions are within 0.5 mm of the requested physical dimensions.
6. Exact self-intersection checks pass in production mode.
7. The input measurements are physically calibrated. Manual measurements are calibrated by definition; image-only estimates are never a production PASS.
8. The shipped binary assets match assets/templates/asset_manifest.json.

## Image-only generation

The API can run an image-only job to support evaluation and development. Image scale can be supplied with reference_frame_width_mm, which improves front-view scale estimation.

A single front-frame reference does not by itself prove every physical dimension extracted from side/top images. Therefore the measurement object remains non-production unless all required dimensions have a trusted physical calibration source.

For a production catalog workflow, supply authoritative product dimensions from the manufacturer/catalog system, or a validated per-view scale-calibration method. Do not convert the default 135 mm photographic assumption into a production measurement claim.

## Release command

Run with the real LFS assets installed:

    DEFIRM_PRODUCTION_MODE=1 python scripts/production_release_gate.py

The release gate verifies the binary asset manifest, required template readiness, Open3D availability, and the complete pytest suite.

## Delivery requirement

The GLB and NPZ runtime files are tracked with Git LFS. A deployment must check out LFS content; Git LFS pointer text is not a usable runtime asset.
