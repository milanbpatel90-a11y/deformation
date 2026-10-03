# GT_001 Gold Template

GT_001 is the required default production deformation bundle for the
rectangle-family frame. Runtime deformation uses `geometry/template.glb`,
`deformation/basis.npz`, `deformation/basis_metadata.json`, and the landmarks
in `metadata/landmarks.json`. The NPZ and GLB vertex/face ordering is checked
at load time. Internal geometry and input dimensions use millimetres; the
exported glTF scene uses metres.

## Deformation and supported range

`backend.deformer.basis_deformer.BasisDeformer` measures the actual rest-state
dimensions, differentiates the supplied basis to form a dimension Jacobian,
and applies its inverse to requested frame width, lens width, lens height, and
bridge gap. Lens width/height and bridge regions are isolated by a monotone
regional coordinate field. Temple length is solved from its root/tip chord.
The shared coupled constraint is:

```text
frame_width >= 2 * lens_width + bridge_width + 2 * rim_thickness
```

Per-field range defaults are declared in `deformation/basis_metadata.json`.
The API and direct deformation path additionally enforce the coupled formula;
the engine rejects unsupported values rather than clamping them. Measured
geometry must remain within 0.5 mm of the requested dimensional controls.

## Validation

Run these from the repository root:

```powershell
python -m pytest -q --tb=short
python -m scripts.check_readiness
python -m scripts.validate_production_glb
python -m scripts.audit_glb output/glb-audit/standard.glb --expected-width-mm 135 --tolerance-mm 0.01 --output output/glb-audit/inspection.json
```

Tests check the real bundle, its calibration response, source immutability,
coupled input rejection, symmetry, positive deformation Jacobian, and selected
extreme cases for triangle self-intersection. The exported GLB is checked
independently for metre scale, dimensions, primitive attributes, normals,
indices, and degeneracy. Exact per-job intersection checks are enabled with
`DEFIRM_PRODUCTION_MODE=1`; this mode fails closed if Open3D is unavailable or
the geometry check fails. Use development outputs for review and do not mark a
`REVIEW` result as release-ready.

Numerical agreement is not evidence that an unreferenced product image has a
known physical scale, nor does a rectangular template reproduce unrelated
silhouettes. Real-product image validation is tracked separately in
`dataset/real_products/README.md`.

## Asset delivery

The runtime GLB and basis are stored with Git LFS. Install Git LFS and fetch
the assets when cloning (`git lfs install`, then `git lfs pull`). A metadata-
only checkout cannot run this template. The Rhino authoring file is not needed
at runtime.
