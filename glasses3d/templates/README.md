# Template assets

Templates can be supplied as `.obj` or `.glb` files and registered in `pipeline/template.py`. Each imported asset should expose meshes named `front`, `lens_L`, `lens_R`, `temple_L`, and `temple_R`, with hinge pivots at the temple starts and the bridge center as scene origin. The default implementation uses a procedural fallback so the pipeline runs without external assets.
