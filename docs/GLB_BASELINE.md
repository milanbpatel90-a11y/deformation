# GLB baseline, before export repairs

Checkpoint: `a2bcc4bc`, branch `codex/recoverable-appearance-pipeline`.
Working branch: `codex/glb-production-audit`.

Actual user export: Downloads/f03c2ce508ca41ef9886e53efd0bb10a.glb.
Canonical production geometry: assets/templates/GT_001/geometry/template.glb.
No synthetic geometry is used for this audit. Detailed binary/mesh inventory is in glb-audit-baseline.json.

- User GLB: 3,621,432 bytes, 185,686 vertices, 162,304 triangles, 13 independently named components/primitives, 2 materials, no textures or UVs.
- glTF 2.0, asset.generator is Trimesh. Project generator typo occurs in scene extras, not the third-party asset generator.
- World extents: 140.44920349 x 49.46155262 x 146.86434889 coordinate units. Deformation parameters and basis are explicitly mm; export has no 0.001 conversion. Under glTF these are metres, a confirmed 1000x unit error. There is no existing metres label to correct.
- Requested frame width is 135 mm; actual Frame X extent is 140.44920349 mm. This is a separate dimensional error, not a unit issue.
- Source template Frame width is 145.33226013 mm versus nominal 140.0; measured lens box is 55.75033617 x 36.52241039 mm versus nominal 55.79 x 36.92. Bridge and temple definitions use landmarks rather than global bounding boxes.
- Each sibling has one -90 degree X matrix; no nested repeated conversion. This legitimately maps source Z-up to viewer Y-up.
- All primitives omit NORMAL. No UVs are needed for current untextured PBR materials.
- Scene extras contains extras.defirmation plus two identical anchor maps. All component nodes are siblings; temple origins are world origin, not hinge pivots.
- No unused vertices, duplicate index triangles, or zero-area triangles were detected. Many split vertices create apparent boundaries; they need a welded diagnostic before being called holes. Right-side components have opposite signed volume; closed-component winding needs independent confirmation before repair.
- Temple joints: 37,876 vertices / 18,944 triangles each. Dense/split data, not proof that surfaces may be deleted.
- Original inner inserts extend to native Y=142.887 mm while shortened temples end at Y=129.674 mm. The checkpoint already transfers temple displacement to inserts; source assets remain unchanged.
- Source scale_config.json explicitly states physical reference is assumed, not measured from a real product. Absolute dimensions still depend on user-provided mm values.

Export path: FastAPI backend/api/main.py -> backend/pipeline/deformation_pipeline.py -> backend/exporter/glb_exporter.py. CLI uses the same pipeline. Gold uses basis_deformer.py; legacy uses engine.py. Blender preparation scripts and glasses3d have separate export paths and were inspected; they are not called by this runtime and are not evidence that this GLB is valid. Blender normalization currently treats objects separately and fix_glb.py assigns names by vertex counts; neither should be run on the registered basis asset.
