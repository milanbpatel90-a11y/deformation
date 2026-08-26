# Binary assets pending local import

The Gold Template source package contains binary/runtime assets that should be stored with the template:

- `geometry/Gold_Template_Runtime.glb` -> `geometry/template.glb`
- `deformation/basis.npz`
- `deformation/_unified_faces.npy`
- `deformation/_unified_verts.npy`
- `geometry/Gold_Template_Rhino_v1_scaled.3dm`
- `rhino/Gold_Template_Rhino_v1_scaled.3dm`
- validation image output

The repository integration branch contains the registry/loader and the canonical metadata contract, but the GitHub connector cannot push these local binary files directly.

Run:

```bash
python scripts/install_gold_template.py /path/to/Gold_Template.zip
```

Then verify:

```bash
python scripts/validate_gt001.py
```

For Git, use Git LFS for `.glb`, `.npz`, `.npy`, and `.3dm` if the repository policy requires it.
