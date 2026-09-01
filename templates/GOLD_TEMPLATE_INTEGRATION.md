# gold_template integration notes

## What this is
A new template added to this repo's existing template system (`templates/`,
`backend/deformer/`), built from the Gold Template geometry developed and
geometrically validated in a separate chat session (Rhino source -> scale
correction -> harmonic lens/rim/bridge coupling -> boundary/self-intersection
testing). That session's full package (basis.npz, its own deformation_engine.py,
tests, validation reports) is a DIFFERENT format from this repo's own
`backend/deformer/*` system - the two don't share code. This integration
takes that session's validated GLB geometry and adapts it to speak this
repo's actual descriptor schema, so this repo's existing (more mature,
already-70-80%-built) pipeline can use it.

## Files added
- `templates/gold_template.glb` - the template geometry, 16 mesh nodes
  (13 original parts + Bridge/LeftRim/RightRim extracted as real sub-meshes
  of Frame, see below) + 5 empty nodes (Bridge_Center, Left_Hinge,
  Right_Hinge, Left_Temple_End, Right_Temple_End) at exact, geometrically
  verified landmark positions.
- `templates/gold_template.json` - top-level dimensions metadata
  (frame_width=140, lens_width=55.79, lens_height=36.92, bridge_width=17.9,
  temple_length=155.02mm - all measured from the actual mesh, not guessed).
- `templates/descriptors/gold_template.json` - the deformer descriptor,
  matching `DescriptorLoader.REQUIRED_KEYS` exactly.
- `templates/templates.json` - registered the new template (thumbnail_path
  points at a PNG that does not exist yet - not required by anything tested
  here, but worth generating before this ships in a UI template picker).

## What was actually verified, not just asserted
I ran this repo's REAL code against these files, not just checked schema
shape:

1. `DescriptorLoader().load("gold_template", ...)` - loads successfully.
2. Found and fixed a real bug in the process: `mesh_aliases` only resolves
   inside the loader's internal `geometry` dict (used for pivot/measurement
   inference) - it does NOT propagate into `DeformationContext.meshes`,
   which is built straight from the GLB's raw `scene.geometry` keys. My
   first attempt (aliasing Bridge/LeftRim/RightRim -> Frame, following the
   same pattern `geometric_metal.json` already uses in this repo) loaded
   fine but raised `KeyError: 'Mesh LeftRim not found'` the moment I
   actually ran `MeshDeformer.deform()`. Fixed by extracting Bridge/
   LeftRim/RightRim as real, separately-named sub-meshes of Frame, using
   the exact vertex-index region masks already computed and validated for
   this geometry (not a new guess - see the originating session's
   `masks.json` region_masks).
   **This same bug likely affects `geometric_metal.json`'s use of
   mesh_aliases too - worth checking if that template is still blocked.**
3. Found and fixed a second real bug: `_load_lens_planes` defaults
   `height_mm` to `geom.extents[1]`, which assumes a Y-up axis convention.
   This template's geometry convention (from its own build spec) is
   X=left/right, Y=front/back(depth), Z=up/down - so the default silently
   read the lens's 2.9mm depth as its "height" instead of the real 36.9mm.
   Fixed via the `aspect_width_mm`/`aspect_height_mm` override fields the
   loader already supports (no repo code changed).
4. Ran the full pipeline end-to-end:
   `MeshDeformer(scene, dims).deform(context)` through all 7 stages
   (RimDeformer, BridgeDeformer, TempleDeformer, LensDeformer,
   ConstraintSolver, SymmetrySolver, MeshSmoother) - **succeeded**:
   `QualityReport(score=89.5, passed=True, breakdown={'geometry': 100.0,
   'constraints': 85.0, 'symmetry': 85.0})`.

## What was NOT verified (be aware before treating this as done)
- constraints=85 and symmetry=85 (not 100) in that quality report - I did
  not dig into why; worth checking `quality_checker.py`'s scoring to see
  what's costing 15 points on each before assuming this is production-fit.
- Only the lens-plane axis convention was checked for axis-order bugs.
  `bridge_deformer.py`, `rim_deformer.py`, `temple_deformer.py`'s internal
  math were exercised (they ran without crashing, produced a passing
  score) but not independently audited line-by-line for other axis-order
  assumptions the way the lens plane one was caught.
- No thumbnail image generated for `templates/thumbnails/gold_template.png`.
- Not tested against the actual image-upload -> measurement -> template
  pipeline (`backend/pipeline`, `backend/api`) - only the deformer stage
  directly, with hand-built `Measurements`/`TemplateInfo` objects.
- I do not have GitHub push access in this environment - these files are
  staged locally; see the commands below to actually get them into the repo.

## Commands to actually push this
```bash
git checkout -b add-gold-template
git add templates/gold_template.glb templates/gold_template.json \
        templates/descriptors/gold_template.json templates/templates.json
git commit -m "Add gold_template: scale-corrected, harmonically-coupled eyewear template"
git push origin add-gold-template
# then open a PR, or push directly to main if that's your workflow
```
