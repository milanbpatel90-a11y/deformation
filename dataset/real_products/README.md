# Real-product deformation benchmark

This is the labeled evaluation contract for real glasses. There are no product
records in this folder yet: generated examples and the template source models
are not ground-truth product validation. A release benchmark must use at least
100 distinct products with dimensions measured from verified CAD or calipers.

Create a `manifest.json` here using the following shape:

```json
{
  "schema_version": 1,
  "acceptance_criteria": {
    "max_mae_mm": 1.0,
    "max_error_mm": 3.0
  },
  "products": [
    {
      "id": "sku-0001",
      "measurement_source": "verified CAD drawing revision A",
      "expected_template": "RB_001",
      "images": {
        "front": "sku-0001/front.jpg",
        "three_quarter": "sku-0001/three-quarter.jpg",
        "side": "sku-0001/side.jpg"
      },
      "measurements_mm": {
        "frame_width": 135.0,
        "lens_width": 52.0,
        "lens_height": 40.0,
        "bridge_width": 18.0,
        "temple_length": 140.0,
        "rim_thickness": 2.0
      }
    }
  ]
}
```

The numerical acceptance limits above are examples that the product owner must
approve for the target use. Do not copy example dimensions into the real set.
Use the supported template IDs from `/api/templates`. Capture images with the
same view definitions and plain backgrounds, and record one known physical
scale reference if photo-derived dimensions are expected to be metric.

Run from the repository root:

```powershell
python -m scripts.benchmark_real_products dataset/real_products/manifest.json
```

The report records per-product measurement errors, template selection, output
acceptance, MAE, RMSE, worst error, and failures. The gate fails when the
manifest has fewer than 100 products, acceptance criteria are missing or
exceeded, template selection is wrong, any job fails, or an output is only
marked `REVIEW`. Run exact geometry acceptance with `DEFIRM_PRODUCTION_MODE=1`
for that stage. Product-image quality and coverage still require human review.

Do not commit private customer images or proprietary CAD unless their owner has
approved storage in this repository. The harness accepts a manifest stored
outside Git as well.
