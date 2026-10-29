# Real eyewear segmentation dataset integration

This integration adds a reproducible import and QA workflow for `eyewear_segmentation_1000_aiqa` without replacing the repository's current segmentation model, synthetic dataset, or deformation implementation.

## Dataset status and limitations

- 1,000 source images: 700 train, 152 validation, 148 test; product groups were separated across splits in the source manifest.
- 1,000 binary masks and 1,000 YOLO polygon text labels, class `0: eyewear`.
- All 1,000 annotations are marked `AI_PRELABEL_NOT_HUMAN_VERIFIED`. These are automatic object-envelope prelabels, not gold-standard segmentation.
- YOLO's simple polygon instance format does not naturally encode lens openings as holes. A filled whole-object envelope can be correct only when that is the intended class definition. If the model needs to distinguish frame pixels from lens material or background, use part-specific classes and a representation that supports holes or multiple masks as appropriate.
- The source card identifies Hugging Face `shravya11/eyeglasses-dataset`, pinned revision `7011dd3ac628040a8885a4b597399409643a9e04`; its metadata reports a CC-BY-4.0 licence tag. Preserve attribution and confirm upstream terms before commercial use.

## Import on Windows PowerShell

1. Extract `eyewear_segmentation_1000_aiqa.zip) outside the repository. Do not copy its `.venv`.
2. From the repository root, run:

```powershell
python import_eyewear_segmentation_dataset.py `
  --source "C:\path\to\eyewear_segmentation_1000_aiqa" `
  --repo-root .
```

Default destination is `dataset/real_eyewear_1000`. If it exists, the importer refuses to overwrite it. Back up first before explicitly passing `--replace`.

## Validate after importing

```powershell
python validate_eyewear_dataset.py --dataset-root dataset/real_eyewear_1000 --check-mask-dimensions --json output/real_eyewear_1000_validation.json
```

The validator checks manifest references, file presence, label syntax/class ID, normalized coordinates, and optionally image/mask dimensions. **It cannot assess visual accuracy.** Review the mask over the source image in the review tool or CVAT/Label Studio.

## Human review and application

1. Open `dataset/real_eyewear_1000/HUMAN_REVIEW_TOOL.html` in a desktop browser, pick the dataset root, and review image/mask pairs.
2. Record approved, corrected, or rejected decisions with a reviewer ID. Export audit JSON and CSV frequently; browser-session persistence is not a substitute for exported backups.
3. For corrected masks, save each downloaded PNG into a `review_masks` folder, keeping the expected `<image_id>_mask.png` filename.
4. Run the helper in dry-run mode from the dataset folder:

```powershell
python apply_reviewed_masks.py --dataset . --review-masks .\review_masks --audit-json .\human_review_audit.json
```

5. Inspect reported errors and back up the dataset. Then apply:

```powershell
python apply_reviewed_masks.py --dataset . --review-masks .\review_masks --audit-json .\human_review_audit.json --apply
```

The helper backs up replaced masks under `human_review_backups/` and regenerates matching YOLO polygons. Re-run the validator afterward and visually spot-check corrected labels before training.

## Training integration

The existing `train_yolov8_seg.py` expects a `splits/manifest.json` layout for its validation command, whereas this dataset uses `manifest.csv` with `images/{train,val,test}`, `masks/{train,val,test}`, and `labels/{train,val,test}`. Do not run that existing validation command against this dataset as-is. Use the new validator, then point Ultralytics to a portable YAML config for this dataset. Keep `test` out of training and make final performance claims only against human-verified labels.

## Repository integration policy

- Dataset image/mask/label binaries are not committed by the importer; it imports them locally from an extracted dataset.
- Existing `dataset/`, `models/`, `runs/`, `backend/segmentation/segmenter.py`, and model weights remain untouched.
- Keep the real dataset separate from the existing synthetic dataset until review and controlled real-image validation succeed.
