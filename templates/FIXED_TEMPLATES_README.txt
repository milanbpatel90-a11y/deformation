Defirmation - 5 templates + engine fixes (topology-cleaned 2026-10-20)

Copy the folders in this zip over your project root (same relative paths).

templates/   cat_eye_gold, slim_rectangle_steel, rectangle_metal, round_wire_metal, oversized_clear (.glb + .json)
templates/descriptors/   matching descriptor JSONs (version 1.1)
backend/     deformation_pipeline.py (alias axis-bake fix + working fallback), deformation_context.py, lens_deformer.py, rim_deformer.py
tests/       updated tests
changes.patch  full code diff (git apply) if you prefer patching

Already pushed to github main.
Note: *.glb is in .gitignore; use `git add -f templates/*.glb`.

=== Fixes applied 2026-10-20 ===

Topology:
- Removed degenerate / tiny connected components (<4 faces) from all 5 GLBs.
- cat_eye_gold: Left/RightRim 21→2 comps; Bridge 55→3; temples cleaned.
- oversized_clear: Left/RightRim 16→1 comp each (single clean loops).
- round_wire_metal: LeftRim 22→1, RightRim 22→1–2; Bridge cleaned.
- rectangle_metal / slim_rectangle_steel: rims reduced; remaining multi-comps on
  Frame/Hinge/NosePad are intentional CAD shells (front/back), not garbage.

Metadata:
- Root JSON notes rewritten (removed outdated topology warnings, documented
  intentional narrow lens height 20.5 mm and oversized scale assumption).
- Descriptors bumped to version 1.1; template_dims synced; processing stamp added.
- round_wire_metal hinge confidence raised 0.6 → 0.75 (temple pivots are reliable).

Remaining known characteristics (not defects):
- rectangle_metal & slim_rectangle_steel: lens_height=20.5 mm (source feature).
- oversized_clear: absolute scale anchored at lens_width=50 mm (no product ref).
- cat_eye_gold hinges: minimal marker geometry; use temple pivots for deformation.
- High-poly templates retain multi-shell CAD structure on hinges/nose pads.
