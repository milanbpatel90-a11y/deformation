# Ground-truth measurement benchmark

The checked-in samples.json contains three synthetic plumbing samples so the benchmark path has deterministic fixtures. They are not an accuracy baseline. Replace or extend samples.json with 15–25+ real eyewear samples measured in millimetres before claiming accuracy.

Required fields per sample:
- id, image, archetype, source, measured_by
- ground_truth.frame_width
- ground_truth.bridge_width
- ground_truth.lens_width
- ground_truth.lens_height
- ground_truth.temple_length
- matching per-field tolerance

The loader rejects unknown fields and any units other than mm.

After the first trustworthy benchmark, copy its overall metrics into baseline.json. Do not use synthetic images to establish the production baseline.
