# Production GLB Template Requirements

The deformation engine only selects templates that can be deformed safely. A template is not production-ready merely because it opens in a 3D viewer.

## Required independently addressable meshes

Every production template must contain eight distinct logical mesh components:

- `Frame`
- `Bridge`
- `LeftRim`
- `RightRim`
- `LeftLens`
- `RightLens`
- `LeftTemple`
- `RightTemple`

Each logical component must resolve to a different source mesh. Mapping multiple logical parts to one combined mesh is rejected. Duplicating one combined mesh under several names does not make the components independently deformable.

## Explicit mesh mapping

If the source GLB uses different mesh names, add reviewed mappings to the matching descriptor:

```json
{
  "mesh_aliases": {
    "Frame": "Frame_Main",
    "Bridge": "Bridge_Mesh",
    "LeftRim": "Rim_Left",
    "RightRim": "Rim_Right",
    "LeftLens": "Lens_Left",
    "RightLens": "Lens_Right",
    "LeftTemple": "Temple_Left",
    "RightTemple": "Temple_Right"
  }
}
```

Replace the example source names with the exact mesh names in the GLB. Do not rely on automatic vertex-count or centroid guessing. If the mapping is ambiguous, the template will be rejected.

## Coordinate and measurement contract

- Template geometry and deformation measurements use millimetres internally.
- Physical dimensions must come from user measurements or manufacturer specifications for production.
- Image-only estimates are marked uncalibrated and production generation rejects them by default.
- Exported GLB geometry and embedded VTO anchors are converted to glTF metres.
- The exported GLB is reopened and checked before the API reports success.

## Required checks before marking a template ready

1. Confirm all eight required components exist and are independently addressable.
2. Confirm the descriptor maps each logical component to the correct physical mesh.
3. Confirm the source mesh names and node transforms are unambiguous; instanced geometry with differing transforms must be split.
4. Record authoritative physical dimensions and verify that template metadata uses millimetres.
5. Run descriptor, deformation, symmetry, and GLB export regression tests.
6. Generate from real product photos plus known physical measurements.
7. Reopen the exact exported GLB in the browser viewer and verify dimensions, materials, bridge/rim/lens/temple geometry, and face placement.
8. Do not mark a template production-ready based only on successful parsing or synthetic tests.

The API template matcher filters out templates that fail these checks. If no valid template remains, generation fails closed instead of silently substituting another template.
