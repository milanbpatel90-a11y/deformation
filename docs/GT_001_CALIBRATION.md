# GT_001 physical deformation calibration

This change is limited to deformation, coupled API validation, readiness and tests. `backend/exporter/` and mesh reduction code are unchanged. Source `template.glb` and `basis.npz` are never rewritten or replaced.

## Root cause and calibration math

The previous pipeline used nominal defaults rather than physical measurements, then scaled the complete X axis to enforce frame width. This changed lens widths a second time. Its broad Frame basis also coupled frame width and lens width into the bridge gap.

Measured native rest dimensions, in order FW, LW, LH, BW:

```text
m0 = [145.33226013, 55.75033617, 36.52241039, 13.11630917] mm
```

Bridge width is now explicitly the horizontal gap between the left and right lens bounding boxes. Temple length retains the delivery's root-to-tip chord definition, not curved arm arc length.

The derivative is computed from the actual NPZ data with a 0.0001 mm coefficient step:

```text
J[i,j] = (measure(V0 + B[j]*h)[i] - measure(V0)[i]) / h

J = [[1.03808757,  0.07388114, 0,          0.13650216],
     [0,           0.99928905, 0,          0],
     [0,           0,          0.98923105, 0],
     [1,          -1.05074276, 0,          1]]

coefficients = inverse(J) @ (requested_mm - measured_rest_mm)
```

For example, the isolated lens-width factor is **1.00071145** and lens-height factor is **1.01088618**. The full inverse removes cross-coupling rather than merely multiplying all axes by a common scalar:

```text
inverse(J) ~=
[[ 1.10915725, -0.24120221, 0,          -0.15140236],
 [ 0,           1.00071145, 0,           0],
 [ 0,           0,          1.01088618,  0],
 [-1.10915725,  1.29269253, 0,           1.15140236]]
```

The calibrated NPZ coefficients are applied to the optical regions to obtain their target bounds. The unsafe broad frame displacement is replaced by an ordered regional map through measured source frame/lens boundaries and calibrated target boundaries. These are computed from geometry and input dimensions; there are no hardcoded per-product vertex offsets.

Lens regions remain affine. Lens width cannot move the bridge interval. Frame width affects the outer margin intervals and does not rescale the lenses. Bridge width translates the lenses apart while preserving their widths, which is physically necessary when changing the gap. Height has a unit mask over the lenses, a zero mask in the bridge core and smooth transitions through the rim junctions. Neighboring components share the same coordinate field, preserving attachment positions.

## Anti-fold behavior

Transition intervals use:

```text
q(t) = (1-a)*t + a*(3*t*t - 2*t*t*t)
q'(t) = 1-a + 6*a*t*(1-t)
a = 0.4 * smoothstep(abs(interval_scale - 1))
```

Thus 0 <= a <= 0.4 and q' >= 0.6 inside an interval. Knots remain exact, interpolation stays bounded, and outside continuation has positive slope. Damping depends on the local interval strain; a temple edit cannot affect frame interpolation.

The full deformation is triangular: X=f(x), Z=g(x,z), Y=h(y). The X and Y maps are increasing, and the Z derivative is 1+w*(height_scale-1)>0. The resulting Jacobian determinant is bounded positively. This rules out foldovers in the continuous field; separate exact triangle-intersection tests check the actual piecewise-linear meshes at extreme valid inputs.

The height falloff spans the measured outer rim band. A narrower cutoff at the inner temple wall created excessive curvature for long artist triangles; that was identified with surface-orientation diagnostics and corrected without weakening dimensional tests.

Temple motion begins behind the complete fixed frame/hinge assembly. The destination tip is solved from TL^2=dx^2+dy^2+dz^2. Both the outer arm and its insert use the same longitudinal map. The frame, lenses and fixed hinge housings do not move when only temple length changes.

## Coupled input contract

`backend/template_library/compatibility.py` implements one shared formula:

```text
FW >= 2*LW + BW + 2*RT
LW_max = (FW - BW - 2*RT)/2
BW_max = FW - 2*LW - 2*RT
RT_max = (FW - 2*LW - BW)/2
FW_min = 2*LW + BW + 2*RT
```

These dependent intervals are intersected with the installed template's independent bounds. They describe changing one field while keeping the others fixed; `feasible:false` identifies an empty interval. Manual generation never clamps a requested value.

All three generation endpoints reject coupled invalid values before image processing with HTTP 422 and `{detail, ranges}`. Successful generation includes dependent `ranges`. `POST /api/measurements/ranges` takes a Measurements JSON object and returns the current intervals without generating a mesh.

Image suggestions explicitly adjust estimates to a feasible combination and explain the changes in `adjustments`. They expose `dependent_ranges`; the existing `ranges` field remains the static template limits so an existing viewer does not incorrectly freeze dependent bounds after the user edits another field. The geometry engine validates the same formula, protecting CLI/direct Python callers as well as HTTP callers.

## Verification

Verified on 2026-09-26: **89 tests passed, zero skipped** (`145.16s`). `scripts.check_readiness` returned `ready: true` for GT_001 and rejected the invalid coupled case. Its four deformation cases had maximum native-coordinate dimension error below 0.000001 mm. The 11-case production sweep passed, preserved both source binary hashes, and produced an 81,580-vertex standard GLB. Khronos validation of that output reported zero errors, warnings, infos or hints. These numerical errors describe mesh bounds, not real-world measurement certainty. Other unprepared templates remain unavailable.

The formerly skipped test was replaced with real `BasisDeformer` integration and its skip decorator removed. Before the fix it failed at a 2.16775 mm lens error and accepted invalid inputs with HTTP 200. The same new assertions then passed; the 0.5 mm tolerance was not relaxed and the test was not skipped again.

Checks include:

- Physical FW/LW/LH/BW and root-to-tip TL, with one-mm independent perturbations.
- Measured-rest identity and unchanged source binary hashes.
- 60 valid corner combinations; 48 invalid combinations rejected.
- Exact Open3D triangle self-intersection checks on Frame, both lenses and both temples at small and large valid extremes. Diagnostic copies weld coincident seam vertices; runtime and source topology remain untouched.
- Surface areas, finite coordinates, positive bounding volumes and positive deformation Jacobian.
- HTTP 422 and corrective ranges on all generation routes, API range calculation, automatic suggestions and unchanged manual measurements.
- Real binary delivery checks: absent files or Git LFS pointers fail clearly; tests never skip or create substitute geometry.
- Existing exporter, units, material, normals, scene graph and pivot tests continue to run unchanged apart from updating old tests that explicitly expected the now-fixed dimensional failure.

The old 120 mm width fixture with 55.79 mm lenses and 17.9 mm bridge is physically invalid under the new contract. Its successful-deformation fixture now uses 48 mm lenses, while invalid combinations are explicitly asserted to reject. The unskipped integration test's dimensional and geometric assertions remain unchanged from their initial failing execution.

Commands from the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest -q --tb=short
.\venv\Scripts\python.exe -m scripts.check_readiness
.\venv\Scripts\python.exe -m scripts.validate_production_glb
```

Open3D is the existing project dependency used for exact triangle checks. Install the project's requirements in the test environment. Real assets must be installed at `assets/templates/GT_001/geometry/template.glb` and `assets/templates/GT_001/deformation/basis.npz`; metadata-only Git checkouts and unresolved LFS pointers are insufficient.

The prior audit remains historical evidence. This calibration supersedes its lens/bridge dimensional and coupled-range blockers. It does not claim photo-identical geometry, manufacturing certification, a calibrated physical scale from unreferenced images, or a new rim cross-section/temple-curvature control. RT currently reserves outer clearance; the artist's authored cross-section is retained.
