# Real eyewear segmentation dataset (local data)

This folder is the reserved integration target for the real-image eyewear segmentation dataset. The import script copies images, masks, labels, manifest, the review helper and source provenance here from the extracted dataset.

The dataset is AI-prelabelled and **not human-verified**. Do not use every annotation as gold standard until review is complete.

## Populate locally

From repository root:

```powershell
python import_eyewear_segmentation_dataset.py --source "C:\path\to\eyewear_segmentation_1000_aiqa" --repo-root .
```

The default importer destination is this directory and refuses to overwrite an existing folder.

After import, run:

```powershell
python validate_eyewear_dataset.py --dataset-root dataset/real_eyewear_1000 --check-mask-dimensions
```

See [EYEWEAR_DATASET_INTEGRATION.md](../../EYEWEAR_DATASET_INTEGRATION.md) for human review, applying corrected masks, limitations of YOLO polygons for holes, and safe training instructions.
