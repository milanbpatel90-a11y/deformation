# Production GLB audit and implemented repairs

## Status

Export/rendering repairs are implemented on `codex/glb-production-audit`.
The previous working pipeline is committed at `a2bcc4bc` on `codex/recoverable-appearance-pipeline`.
**The Gold Template is not certified production-ready for dimensional fitting.** The independent sweep reports that explicitly; successful serialization is not a dimensional pass.

## A. Confirmed root causes

- `backend/exporter/glb_exporter.py:GLBExporter.export`: raw millimetre geometry was written as glTF coordinates without conversion. glTF world coordinates are metres. No actual `meters` metadata label existed in the old file; this was an omitted conversion, not merely a mislabeled field.
- `BasisDeformer.deform`: default frame width was labelled 140 mm but the real Frame was 145.332260 mm. A 135 mm request produced 140.442276 mm for the standard regression case (140.449203 mm in the user's attached export).
- `GLBExporter._attach_metadata`: an extra `extras` wrapper, duplicate anchor maps and project spelling `defirmation` leaked into output. The old glTF asset generator itself correctly named third-party Trimesh and was not a typo.
- Source meshes had no exported normals. Welded diagnostics confirmed closed negative-volume right-side lenses/temples/hinges/joints. Fixing normals on disconnected vertices in the old pipeline did not repair these shells reliably.
- All temples were root siblings, so rotating a temple node rotated it around the global origin. The repeated sibling -90 degree X rotations were legitimate Z-up to Y-up conversions, not a double-rotation bug.
- `BasisDeformer.__init__`: temple inserts did not receive temple-length deformation. The preceding checkpoint fixed their protrusion by transferring the outer temple displacement.
- Viewer PBR metal had no environment illumination, which made valid rose-gold material appear almost black. A bundled Three.js RoomEnvironment was added after reproducing this rendering defect; no external HDR or normal map was introduced.

## B. Changes

- `backend/exporter/glb_exporter.py`: world-consistent metre conversion, explicit input-unit override, idempotent re-export, hinge-relative parents, one canonical anchor map, flattened project metadata, normal export, exact compact indices, matching sidecar and buffer targets.
- `backend/exporter/geometry.py`: export-only coincident-vertex cleanup, closed-shell winding repair, independently verified mirror winding for open inserts, 35-degree shading crease threshold, retained PBR/UV data, world-space measurement helpers.
- `backend/exporter/validation.py`: independent GLB binary/accessor reader and recursive world-transform audit. This does not reuse the exporter scene graph for dimension assertions.
- `backend/deformer/basis_deformer.py`: complete-assembly X calibration to requested outer Frame width, world-space hinge centers, declared mm units and real lens-error quality flags.
- `backend/pipeline/deformation_pipeline.py`: stops overwriting the exporter's canonical sidecar with a second metadata calculation.
- `viewer/index.html`: metre-aware clipping planes, aspect-aware bounding-sphere framing, explicit dimensional-review status and opt-in render diagnostics. Camera mode restores its existing camera-scale convention.
- `tests/test_glb_export_contract.py`: production-asset scale, independent components, normals, materials, unchanged source topology, surface positions, nested transforms, unit idempotency, local hinge rotation, symmetry and deformation independence.
- `tests/test_gold_template.py`: adjusts the rest-state expectation for the confirmed width calibration; explicitly expects residual dimensional errors to fail quality.
- `scripts/audit_glb.py`, `scripts/validate_production_glb.py`, `scripts/validate_gltf.cjs`: repeatable measured audits, source hashes, real-asset sweeps and Khronos validation.

## C. Scale and coordinates

Internal Measurements, source basis, displacements and API measurement values remain mm. Export multiplies local positions by exactly 0.001 and each local node translation by 0.001. Rotation/scale matrices remain unchanged, equivalent to conjugating local transforms by the unit conversion. This works for translated parents as well as the present flat template.

`coordinate_units: m`, `measurement_units: mm`, schema version 2 explicitly separate coordinates from requested dimensions. Sidecar anchors derive from the same canonical metre map used by GLB. Re-imported project output is recognized by `coordinate_units` and not scaled again. For an unrelated metre scene, use `source_units='m'`; the backward-compatible default for untagged pipeline scenes remains mm. No size-based unit guessing is used.

The basis remains Z-up. Its single -90-degree X rotation maps Z-up into Y-up. Mesh attributes retain their local coordinates and all measurements are evaluated through the complete node hierarchy. Left/right names are preserved exactly as authored (Left is positive source X).

The extra X calibration is separate from the 0.001 unit conversion. It scales the complete deformed assembly by requested outer Frame width / measured outer Frame width. It preserves relative component positions, but exposes residual errors in the delivered nominal lens controls; these are measured, not hidden.

## D. Measured before/after

Comparable case: frame 135, lens 56 x 37, bridge 18, temple 130.9, rim 1.2 mm. Before is generated with checkpoint `a2bcc4bc` on the same real asset. The user's earlier export has separate baseline JSON because its lens settings differ.

- Before: 3,498,336 bytes; 185,686 vertices; 162,304 triangles; 13 primitives; 2 materials; Frame width 140.442276 coordinates (intended mm, interpreted as metres).
- After: 2,965,248 bytes; 82,608 vertices; 162,304 triangles; 13 primitives; 2 materials; Frame world width 0.135000005364 m.
- Vertices reduced 55.51%; file reduced 15.24% despite adding normals. Triangle count and component count unchanged. No decimation, Draco, texture maps, component merging or removal of unproven internal surfaces.
- Bidirectional original-vs-exported vertex-surface samples differ by less than 2e-8 m, excluding the separately measured intentional unit/width corrections. This checks both retained and introduced vertices; no geometric simplification is claimed.
- Standard lens after width calibration: 53.792 x 36.602 mm against requested 56 x 37. Actual lens-box gap is 7.686 mm, not the nominal 18 mm bridge control. Physical bridge definition must be calibrated against actual landmarks; nominal controls are not certified dimensions.
- Full low/high sweep: 11 actual deformation cases, all finite and with requested outer frame width; **dimensional readiness fails**. For example, frame-width minimum gives signed lens-box gap -16.133 mm (overlap); lens-width maximum gives -6.389 mm. The supplied parameter ranges are not a validated joint feasible region.

Detailed node matrices, component bounds, primitive attributes and materials: `glb-audit-baseline.json`, `glb-same-input-before.json`, `glb-regression-results.json`.

## E. Hierarchy and component independence

Added `LeftTemplePivot` and `RightTemplePivot`, rooted at the world-space hinge mesh bounding-box centers. Their Y axes are the exported hinge rotation axes. Only the matching temple and inner insert are reparented; fixed hinge/joint housings, frame, lenses and opposite arm stay independent. Child matrices are inverse(pivot world) multiplied by original child world. Tests prove zero rest-position displacement and constant vertex-to-pivot distance under a 60-degree rotation.

Hinge centers are inferred from the real hinge geometry. The delivery landmark called Hinge does not coincide with its hinge mesh center; it was not blindly used. Mechanical CAD axis verification remains outstanding.

Source GLB and basis files are never rewritten. The audit records and rechecks SHA-256 of both. Weld/smoothing/index changes exist only on export copies, after all basis operations. Stable deformation vertex correspondence is preserved.

## F. Tests and validators

- Full pytest suite: 78 passed, 1 skipped before the final nested-parent test addition.
- Focused production-export suite after the addition: 8 passed.
- Final complete suite: **79 passed, 1 skipped** (72.23 seconds).
- Khronos glTF Validator on final real GLB: 0 errors, 0 warnings, 0 infos, 0 hints. Baseline also had 0 structural errors/warnings but 22 buffer-target hints: a structurally valid GLB can still have wrong scale, winding, or dimensions.
- 11-case production dimensional sweep: completed; readiness result false, exit code 2 by design. This is a failed production gate, not a passing regression disguised as success.
- Viewer module syntax check and git diff whitespace check executed. Live image-upload API also returned HTTP 200 with detected metal/rose colour, a measured 0.135 m frame, and quality.passed=false for the unresolved dimensional mismatch.

The one skipped test is an existing legacy-template integration case; it is not substituted for the real Gold Template tests.

## G. Visual and deformation validation

The final GLB loaded in the actual localhost Three.js viewer. Front and oblique views were inspected: transparent lenses, independently visible temples, reflective frame and no console errors. Camera framing was adjusted for the narrow in-app panel and metre scale. This is desktop visual inspection, not a mobile GPU/frame-time certification or photo-identical shape validation.

Actual Three.js counters on both the matching baseline and repaired GLBs: **15 draw calls, 182,784 rendered triangles** (the 162,304 asset triangles plus a second pass for each 10,240-triangle lens). Opt-in `diagnostics=1` shows these counters for repeatable measurement. The closed transparent lenses intentionally retain their existing double-sided BLEND material. Other materials are not made double-sided to conceal winding errors.

## H. Remaining blockers and scope limits

1. **Do not deploy as a dimensionally accurate product generator yet.** The supplied Gold basis lacks a coupled feasible-range solver and accurate lens/bridge calibration. All 11 audited inputs fail the 0.5 mm lens tolerance after width correction. The app exposes `quality.passed=false` instead of presenting a finite-only 100 score.
2. Large bridge deformations are explicitly documented as distorted in the artist's basis metadata. Safely fixing this requires an improved constrained deformation basis/solver with boundary-fit validation; stretching individual parts or editing metadata cannot establish correctness. The stored example outputs are review artifacts, not manufacturing-ready assets.
3. Image shape/material inference remains heuristic. A single rectangular template cannot reproduce geometric, round, rimless and patterned frames precisely. Exact physical mm requires user dimensions or a scale reference.
4. Source scale_config declares its initial physical reference was assumed. Requested outer width is now enforced, but that does not authenticate source product dimensions.
5. Inserts still have open boundaries after welding. No holes were filled or internal surfaces deleted without evidence. CAD hinge-axis accuracy, full-range lens/rim fit, rim-thickness control and temple curve fidelity remain unverified.
6. No arbitrary triangle reduction: topology-changing simplification was deferred because there is no certified full-range deformation/visual error envelope. Lossless export optimizations were measured instead.
7. Blender factory/add-on exports and legacy `fix_glb.py` were inspected but not executed; Blender-specific behavior is unverified. They are separate preparation paths and not used by the runtime repaired here. Do not run the old destructive vertex-count-based splitter against the registered basis asset.
8. No webcam permission was requested and no live-camera fit test was performed. CDN availability, deployment hardening, mobile frame time, memory budgets and device rendering need release testing. Original assets are locally available but Git-ignored; deployment must provision their verified files separately.
9. Schema v2 removes the misspelled/nested project metadata wrapper. Consumers that explicitly read `scene.extras.extras.defirmation` must migrate to `scene.extras.deformation`. Measurement response keys stay unchanged; anchor coordinates now explicitly use metres. Unrelated custom metadata is preserved.

## I. Reproduce

Run from the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest -q --tb=short
.\venv\Scripts\python.exe -m scripts.validate_production_glb
# Exit 2 currently means the measured dimensional production gate failed.
.\venv\Scripts\python.exe -m scripts.audit_glb output/glb-audit/standard.glb --expected-width-mm 135 --tolerance-mm 0.01 --output output/glb-audit/inspection.json
npm install --prefix output/gltf-validation --no-audit --no-fund gltf-validator@2.0.0-dev.3.10
node scripts/validate_gltf.cjs output/glb-audit/standard.glb
Copy-Item output/glb-audit/standard.glb output/audit-production.glb
.\venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Viewer: `http://127.0.0.1:8000/viewer/?glb=/api/output/audit-production.glb&diagnostics=1`.

Specification references: [glTF 2.0 coordinates and attributes](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html), [Khronos validator](https://github.com/KhronosGroup/glTF-Validator).
