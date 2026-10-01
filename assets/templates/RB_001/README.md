# Ray-Ban Wayfarer template (RB_001)

This is a runtime deformation bundle built from `source/ray_ban_glasses.3dm` and registered beside Gold (`GT_001`). The source declares millimetres, but the front spans about 1,914 source units. The importer uses a recorded scale factor of `0.08`, producing a 153.12 mm front, 57.90 mm lenses, 50.04 mm lens height, 24.63 mm bridge gap, and 149.98 mm temple length. This factor is an eyewear-domain inference and should be checked against a known product measurement before claiming physical calibration.

The 3DM has two generic `defaultMaterial` mesh objects and no named empty/point landmarks. Object 0 contains the front and two long temples. Object 1 contains four lens-shell patches (paired by side) and nose-pad components. The importer names those regions from connected-component bounds and left/right position; it derives optical landmarks from lens bounds, temple root/tip points from the temple ends, and hinge pivots from the temple roots because the file contains no separate hinge geometry.

## Runtime files

- `geometry/template.glb`: seven named parts: `Frame`, `LeftLens`, `RightLens`, `LeftTemple`, `RightTemple`, `NosePadLeft`, and `NosePadRight`.
- `deformation/basis.npz`: unified rest vertices, the five-column deformation basis, faces, and matching part order.
- `deformation/basis_metadata.json`: supported parameters and units.
- `metadata/template.json`: template identity, shape/material profile, dimensions, source and landmark method.
- `metadata/landmarks.json`: standard `LM_*` optical, bridge, temple and pivot landmarks.
- Other metadata JSON files record measurements, constraints, topology, vertex/region masks, scale assumptions, and source-object-to-runtime-name mappings.
- `source/ray_ban_glasses.3dm`: unchanged authoring file.

Regenerate from the supplied source with:

```powershell
python scripts/import_rhino_template.py C:\Users\Petpooja-607\Downloads\ray_ban_glasses.3dm
```

The converter needs `rhino3dm` from `requirements-dev.txt`; runtime reads only the GLB, NPZ and JSON files. Binary template assets and the original 3DM are tracked through Git LFS.
