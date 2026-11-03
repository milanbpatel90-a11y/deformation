# Real-world eyewear segmentation: data, labels and training

## Decision: do not treat classification photos as segmentation labels

The repository's existing synthetic dataset/data.yaml is a one-class silhouette dataset (eyewear). The separate imported 1,000-photo dataset is also a one-class whole-eyewear mask set, and its masks were marked AI-prelabelled/not human-verified. Neither dataset can be relabelled as six component classes by changing YAML.

The six-class production contract used by the deformation measurement path is:

| ID | Exact label | Annotation target |
|---:|---|---|
| 0 | eyewear_rim | Frame rims around both lens openings; do not fill the lens interiors |
| 1 | eyewear_temple | Left/right side arms (temples), only where visible |
| 2 | bridge | Bridge joining the two eyewires |
| 3 | left_lens | Left lens area/outline, separated from the frame |
| 4 | right_lens | Right lens area/outline, separated from the frame |
| 5 | nose_pad | Nose pads, when present and visible |

The class order is part of the model API. Do not reorder IDs without changing the segmenter, measurement converter, training YAML and tests together.

## Real product-photo source

The public Hugging Face **Eyeglasses vs Sunglasses Dataset** is a useful real-product image pool:

- 21,766 images, organised into eyeglasses and sunglasses categories.
- Product models are grouped into train and validation splits to reduce product-identity leakage.
- The dataset card declares CC-BY-4.0.
- Its labels are product-category labels only. It does **not** provide rim/temple/bridge/left-lens/right-lens/nose-pad masks.

Source: https://huggingface.co/datasets/shravya11/eyeglasses-dataset

Stream a small, balanced photo subset instead of downloading the full multi-gigabyte dataset:

~~~powershell
pip install -r requirements-real-eyewear-data.txt
python scripts/fetch_real_eyewear_photos.py --split train --max-per-class 250
python scripts/fetch_real_eyewear_photos.py --split validation --max-per-class 100
~~~

The script writes source label, split, SHA-256 and attribution metadata. Images are saved as **UNLABELLED** for segmentation. It deliberately does not create YOLO masks from the eyeglasses/sunglasses category labels.

Additional public mask resources can help bootstrap annotation, but they are not a substitute for the product-photo six-part dataset:

- Roboflow's eyeglass v7 instance-segmentation export declares CC-BY-4.0 and has two labels (glass, frame) across 76 images in that version: https://universe.roboflow.com/azaduni/eyeglass-6wu5y
- Roboflow's cropped-face glasses segmentation v2 declares CC-BY-4.0 and has 785 images with two classes (0, glasses-lenses): https://universe.roboflow.com/yair-etkes-iy1bq/glasses-segmentation-cropped-faces
- The glasses-detector project documents separate segmentation tasks for full glasses, frames, legs and lenses. It aggregates several datasets and is useful as an annotation/reference tool; its labels do not provide all six required component classes: https://github.com/mantasu/glasses-detector

Respect each source's licence and attribution. Do not redistribute source images into this repository unless redistribution rights are confirmed.

## Annotation requirements

1. Use product images, not only face crops. Include isolated front, 3/4, side and top views when available.
2. Keep all photos of the same physical product in one split. Do not let near-identical product shots leak between train, validation and test.
3. Use CVAT or X-AnyLabeling to create six distinct masks. A whole-object SAM/YOLO mask is only an eyewear silhouette; it is not evidence for a rim, lens or temple mask.
4. Review every prelabel. Thin metal rims, transparent lenses, clear acetate, tortoiseshell, nose pads, hinge regions, reflections and dark backgrounds need special attention.
5. If a part is not visible, mark it as occluded/not visible in the review record. Do not draw an imagined mask. A front-only view cannot establish full temple length.
6. Do not use a single filled polygon to represent both the frame and transparent lens openings. Keep lens and rim annotations separate.
7. Retain a reviewer, product/group ID, source URL and licence field in the annotation manifest.

Recommended review manifest columns:

~~~csv
split,local_image,product_id,annotation_status,reviewer
train,images/train/product_001_front.jpg,product_001,HUMAN_VERIFIED_CORRECTED,reviewer_a
val,images/val/product_202_front.jpg,product_202,HUMAN_VERIFIED_ACCEPTED_PRELABEL,reviewer_b
~~~

Accepted status values understood by the training gate are HUMAN_VERIFIED_ACCEPTED_PRELABEL, HUMAN_VERIFIED_CORRECTED, APPROVED and CORRECTED. Rejected, unreviewed or missing statuses are blocked.

## Six-class training

Create a YOLO segmentation folder with images/{train,val,test} and matching labels/{train,val,test}. The data.yaml must declare exactly:

~~~yaml
path: /absolute/path/to/eyewear_parts_dataset
train: images/train
val: images/val
test: images/test
names:
  0: eyewear_rim
  1: eyewear_temple
  2: bridge
  3: left_lens
  4: right_lens
  5: nose_pad
~~~

First run the strict review/label gate:

~~~powershell
python scripts/train_yolo_eyewear_parts.py --data path/to/data.yaml --review-manifest path/to/review_manifest.csv --validate-only
~~~

Only after it passes, train (GPU recommended for 960px masks):

~~~powershell
python scripts/train_yolo_eyewear_parts.py --data path/to/data.yaml --review-manifest path/to/review_manifest.csv --weights yolov8n-seg.pt --epochs 100 --imgsz 960 --batch 4 --device 0
~~~

Use --device cpu for CPU-only systems, but training will be much slower. Keep the test split untouched until the model and thresholds are frozen.

The training gate refuses:
- one-class silhouette YAMLs presented as six-part datasets;
- missing, malformed or out-of-range YOLO polygons;
- unreviewed AI prelabels;
- missing class coverage in train/validation;
- product groups shared across splits.

## Runtime configuration

Set the trained six-class checkpoint explicitly:

~~~text
DEFIRM_YOLO_MODEL=/absolute/path/to/best.pt
DEFIRM_REQUIRE_YOLO=1
DEFIRM_YOLO_CONF=0.25
~~~

The inference module now returns:
- front: combined eyewear silhouette for existing downstream consumers;
- class_masks: masks keyed by numeric class ID;
- part_masks: masks keyed by semantic class name;
- class_names, detection count, confidence threshold and observed-view metadata.

The measurement pipeline reuses those masks from the same inference call. It rejects missing bridge, left/right lens and temple masks rather than fabricating default lens polygons or nominal temple dimensions. A front-only image without a visible temple therefore cannot silently produce a complete 3D measurement set; provide a side/three-quarter product view and reviewed temple mask. The full frame width is measured across all disconnected rim components instead of using only the largest single-eye contour. Rimless designs use the combined lens-plus-bridge envelope for the pixel scale, rather than assuming the rim mask exists.

## What is and is not validated

The repository's CI tests validate the mask/class contract and failure paths. They do not prove model accuracy. A trained checkpoint must still be evaluated on human-reviewed, held-out real products and checked visually in overlays. Report per-class mask precision/recall, mAP50 and mAP50-95, false positives/negatives, and performance by frame type/material. The final test set must never be used for tuning.
