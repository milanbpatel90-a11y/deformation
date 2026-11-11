# GT_002 - NewWayfarer Production

This template registers the supplied `NewWayfarer_production.3dm` as the second deformation-template authoring source and mirrors the existing GT_001 bundle layout.

## Registered source structure

- Rhino source target: `source/NewWayfarer_production.3dm`
- Reference screenshot: `reference/layer_tree.png`
- Components: Frame, left/right rims, Bridge, left/right lenses, temples and hinges
- 21 named landmarks: bridge attachments/center, lens centers/cardinals, hinge/hinge axes, temple roots/tips
- Metadata: parameter schema, constraints, semantic regions, scale config and source manifest

## Runtime status

`runtime_ready` is **false** until an exact `geometry/template.glb`, unified mesh arrays and `deformation/basis.npz` are generated from the Rhino source. These files must not be fabricated from the screenshot or guessed dimensions.

Run `source/export_runtime_from_rhino.py` inside Rhino 8 with the source model open. It validates required named parts/landmarks, records model-space measurements and exports the runtime geometry.

## Production acceptance

Do not mark GT_002 runtime-ready merely because a GLB exports. Require measured dimensions, named-part validation, landmark validation, stable unified topology, valid basis arrays and deformation regression checks.
