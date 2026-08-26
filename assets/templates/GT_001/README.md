# GT_001 Gold Template

This directory is the versioned home for the first deformation-engine Gold Template.

## Runtime contract

A complete GT_001 bundle is expected to contain:

- `geometry/template.glb`
- `metadata/template.json`
- `metadata/parameter_schema.json`
- `metadata/measurements.json`
- `metadata/constraints.json`
- `metadata/topology.json`
- `metadata/masks.json`
- `metadata/region_masks.json`
- `metadata/landmarks.json`
- `metadata/scale_config.json`
- `deformation/basis.npz`
- `deformation/basis_metadata.json`
- `deformation/deformation_engine.py`
- `deformation/_part_order.json`
- validation/test files from the source Gold Template package

The large binary/runtime files are intentionally not committed by this integration patch because the GitHub connector used for this change cannot upload arbitrary binary blobs from the local workspace. Use `scripts/install_gold_template.py` with the supplied `Gold_Template.zip` to populate them locally, then commit/push those files with Git LFS if desired.

## Template identity

- Template ID: `GT_001`
- Schema: `1.1`
- Units: `mm`
- Origin: `LM_BridgeCenter`
- Coordinate system: X=left/right, Y=front/back, Z=up/down

Do not hard-code GT_001 paths in the deformation engine. Load the bundle through the template library.
