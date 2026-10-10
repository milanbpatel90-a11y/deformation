# Component template asset audit

Audit date: 2026-10-09. Measurements come from the packaged GLBs via `scripts/audit_component_templates.py`. GLB node transforms place the geometry in metres; JSON template dimensions are millimetres.

## geometric_metal

The scene has four geometry objects and four nodes. No two nodes share a geometry reference. Every geometry has UV coordinates, vertex normals, and finite coordinates.

| Geometry | Vertices / faces | Connected components | Material | Local extents (m) |
| --- | ---: | ---: | --- | --- |
| Plane_glasses_mat_0 | 61,872 / 96,784 | 530 | glasses_mat | 0.13592 x 0.01051 x 0.04275 |
| Plane.001_glasses_mat_0 | 2,385 / 4,196 | 10 | glasses_mat | 0.01070 x 0.13515 x 0.01202 |
| Plane.002_glasses_mat_0 | 2,385 / 4,196 | 10 | glasses_mat | 0.01070 x 0.13515 x 0.01202 |
| Cube.002_glass_mat_0 | 6,916 / 13,312 | 4 | glass_mat | 0.12305 x 0.00219 x 0.04035 |

The temple objects are independent GLB geometries. The lens geometry contains four disconnected pieces, consistent with two lenses represented by front and back surfaces, but is still one geometry object. The front-frame geometry contains two large connected components whose bounds span the full frame width, including both rims and the bridge. Its many smaller components do not provide a reliable complete partition into Frame, Bridge, LeftRim, and RightRim. Cutting at guessed X coordinates would leave open boundaries and could introduce gaps during deformation.

The former descriptor mapped Frame, Bridge, LeftRim, and RightRim to one object. Those mappings were removed. The template is unavailable for component deformation until an authored split with validated boundary behaviour is supplied.

## rectangle_plastic

The scene has seven geometry objects and seven nodes. Geometry references are not shared. All seven geometries have UV coordinates and normals; all coordinates are finite. The object/node labels are generic or inconsistent with the shapes.

| Geometry | Vertices / faces | Connected components | Material | World extents (m) |
| --- | ---: | ---: | --- | --- |
| Object_0 | 1,910 / 3,824 | 1 | material | 0.1399 x 0.0501 x 0.0145 |
| Object_1 | 1,558 / 1,936 | 64 | material_1 | 0.1462 x 0.0218 x 0.1479 |
| Object_2 | 4,306 / 6,616 | 347 | material_2 | 0.1323 x 0.0069 x 0.0050 |
| Object_3 | 2,493 / 3,648 | 261 | material_2 | 0.1324 x 0.0089 x 0.0236 |
| Object_4 | 2,087 / 3,328 | 144 | material_2 | 0.1329 x 0.0011 x 0.0050 |
| Object_5 | 5,412 / 10,816 | 2 | material_3 | 0.1215 x 0.0431 x 0.0068 |
| Cube | 24 / 12 | 6 | Material | 2.0 x 2.0 x 2.0 |

Frame_Mesh refers to Object_2, but two largest components lie on opposite sides and hundreds of tiny pieces remain. Lens_Left_Mesh and Lens_Right_Mesh refer to objects whose transformed bounds span both sides. Temple_Left refers to a cross-frame-width mesh, while other temple-named objects have different multi-component bounds. Cube is an unrelated two-metre box. No safe one-to-one logical part mapping can be established.

This template is also unavailable for component deformation. No placeholder geometry or inferred aliases were added.

## Reproduction

Run `./venv/Scripts/python.exe scripts/audit_component_templates.py`. The inspection script reports node transforms, local and world bounds, vertex/face counts, connected-component bounds, material, UV and normal data. The GLBs were not modified. Re-enabling either template requires an authored split with independent parts, documented shared-boundary treatment, and round-trip GLB and deformation validation.
