# Five-template parity audit — 2026-10-21

## Outcome

The five ZIP templates are not the same assets as the two reference templates `rectangle_plastic` and `geometric_metal`. The ZIP contains `cat_eye_gold`, `slim_rectangle_steel`, `rectangle_metal`, `round_wire_metal`, and `oversized_clear`, each with a GLB, root metadata JSON, and descriptor JSON.

This branch was created from `main` as an isolated review/change branch. It does not change or overwrite `main`.

## What was checked

- ZIP integrity and inventory: 5 GLB / metadata pairs, 5 descriptors, patch, engine files, and tests.
- All five descriptor JSONs parse and contain the required high-level sections consumed by `DescriptorLoader`.
- Descriptor `template_dims` and root metadata `dimensions` agree on all common dimension fields.
- GLB files load as glTF 2.0 scenes; logical component names are present; no degenerate faces were found using a small-area threshold.
- Topology is not uniformly watertight. This is not automatically disqualifying for eyewear CAD assets, but does mean watertightness cannot be claimed.

## Findings / blockers to a 100% claim

1. The ZIP's root README says commit `626a752` has already been pushed, but the current GitHub `main` commit is `c4cbb0b4478ecb34d4af60b33520fb788eec5a92`. The ZIP changes are not represented by the fetched current versions of the four engine files; I did not overwrite main to force this.
2. The ZIP tests mostly reuse the geometric_metal fixture and the patch does not add a per-template integration matrix proving all five actual GLBs deform correctly.
3. The ZIP's descriptors are structurally complete, but the geometry-backed tests still need to check all five templates through the actual `DescriptorLoader` + `_inject_aliases_into_scene` + full `MeshDeformer` route under native and changed measurements.
4. Mesh component counts and per-template structures differ. The templates should not be considered identical to the two reference templates—only compatible variants if each passes real deformation and export tests.
5. Dimension metadata alone does not prove physical calibration or try-on alignment; independent reference measurements, fit tolerances, and browser-render/export validation are still needed.

## Recommended next validation before merge

- Run descriptor-load and complete deformation smoke tests for each of the five templates using native dimensions (identity-like case) and a changed measurement case.
- Assert finite vertex coordinates, no NaN/Inf, unchanged face/vertex counts, frame dimensions within tolerance, lenses remain fitted inside each rim, left/right temple pivots remain plausible, and GLB export/reload succeeds.
- Add tests that assert actual deformed outputs rather than only mock/synthetic contexts.
- Merge only after these tests pass in CI.
