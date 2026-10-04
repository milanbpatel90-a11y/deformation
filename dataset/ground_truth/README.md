# Ground-truth measurement benchmark

This directory intentionally contains no fabricated measurements or product images. Populate samples.json with 15–25+ real eyewear samples measured in millimetres, then run the benchmark locally with the production pipeline.

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
