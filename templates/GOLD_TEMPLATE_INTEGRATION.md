# Gold Template runtime integration

## Canonical runtime files

- `templates/gold_template.glb` — runtime GLB, reusing the existing repository blob instead of adding a duplicate binary.
- `templates/gold_template.json` — dimensions and named-part metadata consumed by `TemplateLibrary`.
- `templates/descriptors/gold_template.json` — descriptor consumed by the current `backend/deformer` stages.
- `templates/registry.json` — the canonical runtime template registry. Do not introduce a competing `templates/templates.json` registry.

The default is defined once as `DEFAULT_TEMPLATE = "gold_template"` in `backend/pipeline/deformation_pipeline.py`. The CLI, API preferred-template indicator, measurement path, single-image path, and multi-view path use that constant. An explicit template override remains supported.

## Scale and geometry notes

The checked-in runtime model's `Frame` mesh spans approximately **145.332 mm along X**. Earlier metadata stated 140 mm despite the packaged geometry being wider; runtime metadata records the measured X bounds so frame-width scaling is internally consistent. Lens width and height use the actual named lens-mesh bounds. The source package states that no authoritative factory dimension was available, so this scale is a reference calibration, not a manufacturer-certified measurement.

**This is an integration candidate, not a production-certified mesh.** The current GLB was made by extracting logical `Bridge`, `LeftRim`, and `RightRim` meshes from the original `Frame`. A geometry audit found overlapping triangles: the full `Frame` has 40,448 faces, and 40,360 of those triangles also occur in the extracted pieces; `Bridge` also overlaps `LeftRim` and `RightRim` by 4,166 and 4,294 faces respectively. This may cause double surfaces/z-fighting or separate deformers to move coincident geometry differently. Before merging this as production-ready, regenerate genuinely disjoint component meshes and visually inspect the exact exported GLB in the browser viewer.

The descriptor's axis convention is X=left/right, Y=front/back (depth), Z=up/down. Its explicit lens width/height overrides avoid interpreting the roughly 2.9 mm depth as lens height. Other axis-sensitive operations still need real-product validation.

## Validation

```powershell
python -m pytest tests/test_symmetry_solver.py tests/test_symmetry_kdtree.py tests/test_toolkit_registry_schema.py tests/test_gold_template_default.py tests/test_gold_template_deformation.py -q
python -m compileall -q backend tests
```

Passing these tests is necessary but not sufficient for a production release: validate physical scale against a known product, check mesh self-intersections, run the real 4–5-image pipeline, and inspect that exact output in the browser VTO viewer.
