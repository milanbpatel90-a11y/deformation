# Real eyewear segmentation dataset integration

This integration adds a reproducible import and QA workflow for `eyewear_segmentation_1000_aiqa` without replacing the repository's current segmentation model, synthetic dataset, or deformation implementation.

## Dataset status and limitations

- 1,000 source images: 700 train, 152 validation, 148 test; product groups were separated across splits in the source manifest.
- 1,000 binary masks and 1,000 YOLO polygon text labels, class `0: eyewear`.
- All annotations are marked `AI_PRELABEL_NOT_HUMAN_VERIFIED`. These are automatic object-envelope prelabels, not gold-standard segmentation.
- A single filled-object mask is appropriate only when the target is the whole eyewear object. The YOLO polygon format used here cannot accurately encode interior holes like transparent lens openings in a single polygon. If your product needs frame-only geometry or separate frame/lens classes, use part-specific annotations and an appropriate mask format.
- The source card identifies Hugging Face `shravya11/eyeglasses-dataset`, pinned revision `7011dd3ac628040a8885a4b597399409643a9e04`; its metadata reports a CC-BY-4.0 licence tag. Preserve attribution and confirm upstream terms before commercial use.

## Import on Windows PowerShell

1. Extract the dataset ZIP outside the repository. Do not copy the ZIP's Windows `.venv`.
2. From the repository root:

```powershell
python import_eyewear_segmentation_dataset.py `
  --source "C:\path\to\eyewear_segmentation_1000_aiqa" `
  --repo-root .
```

Default destination is `dataset/real_eyewear_1000`. The directory is tracked with its README and `.gitignore`; the importer preserves those code/docs files. It refuses to overwrite existing data directories or data metadata. Back up any existing data before explicitly passing `--replace`.

## Validate

```powershell
python validate_eyewear_dataset.py --dataset-root dataset/real_eyewear_1000 --check-mask-dimensions --json output/real_eyewear_1000_validation.json
```

The validator checks manifest references, file presence, label syntax/class ID, normalized coordinates, and optionally image/mask dimensions. **It cannot assess visual accuracy.** Review every mask against its image in the fixed review tool or CVAT/Label Studio.

## Human review and application

1. Open `dataset/real_eyewear_1000/HUMAN_REVIEW_TOOL_FIXED.html` in a desktop browser, select the dataset root, and compare the green overlay with the original image.
2. Record accepted, corrected, or rejected decisions with a reviewer ID. Export audit JSON and CSV regularly; browser session persistence is not a substitute for exported backups.
3. For corrected masks, download each PNG into a `review_masks` directory, keeping the expected `<image_id>_mask.png` filename.
4. From the dataset folder, run the review helper in dry-run mode first:

```powershell
python apply_reviewed_masks.py --dataset . --review-masks .\review_masks --audit-json .\human_review_audit.json
```

5. Only after checking the dry-run output and backing up the dataset, apply:

```powershell
python apply_reviewed_masks.py --dataset . --review-masks .\review_masks --audit-json .\human_review_audit.json --apply
```

The helper backs up masks/labels under `human_review_backups/`, updates manifest review status, and regenerates YOLO polygons. Polygon conversion uses external contours and cannot represent interior holes; manually check results, especially transparent lenses and thin frames. Re-run the validator afterwards.

## Training integration

The existing `train_yolov8_seg.py` expects a `splits/manifest.json` layout for its validation command, while this dataset uses `manifest.csv` and `images/{train,val,test}`, `masks/{train,val,test}`, `labels/{train,val,test}`. Do not run that existing validation command against this dataset as-is. Use the new validator, then configure Ultralytics against this dataset's portable YAML. Keep `test` out of training and make performance claims only against human-verified labels.

## Repository integration policy

- Dataset images/masks/labels are imported locally and ignored by the target folder's `.gitignore`; this avoids pushing a large unverified binary dataset to Git. The importer preserves the tracked review tools, documentation and helper scripts.
- Existing `dataset/`, `models/`, `runs/`, `backend/segmentation/segmenter.py`, and model weights remain untouched.
- Keep the real dataset separate from the existing synthetic dataset until review and controlled real-image validation succeed.
