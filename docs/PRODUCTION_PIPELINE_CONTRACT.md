# Production pipeline contract

This repository supports one production try-on route:

1. A fine-tuned eyewear segmentation model produces the eyewear silhouette.
2. Shape/material classification and silhouette-based geometry extraction estimate dimensions.
3. Template matching selects a registered template with a GLB, dimensions metadata, and a structurally valid descriptor.
4. The part-aware deformation stages run in millimetres and produce a quality report.
5. Only a passing deformation is exported; the GLB exporter validates triangle data, checks the serialized GLB, and converts coordinates from millimetres to glTF metres.
6. Sidecar measurements are in millimetres; sidecar and embedded anchor coordinates are in metres.

## Segmentation

The primary runtime contract is one class named `eyewear` (eyewear vs. background). The legacy `Auto3DPipeline` may use a validated six-part checkpoint only when its class names and ordering match `MaskToMeasurements`. Generic COCO checkpoints are never selected automatically.

Set `DEFIRM_YOLO_MODEL` to the intended trained checkpoint. If none is available, the API reports `degraded` and uses a low-confidence OpenCV fallback. Empty masks and almost-full-image masks are rejected rather than converted into default measurements.

## Calibration

Product photographs do not provide absolute physical scale by themselves. Pass `frame_width_mm` from manufacturer specifications (100–180 mm) to the image upload endpoints for calibrated frame width. When omitted, the estimator retains a 135 mm reference assumption and marks the returned result as not absolutely calibrated. Lens, bridge, temple, and rim dimensions remain image-derived estimates unless separately calibrated or supplied as measurements.

## Template selection and deformation

Explicitly requested templates are not silently replaced. Auto-selection considers only templates that have a GLB, metadata, and a descriptor containing the required part/constraint sections. Descriptor loading and deformation quality failures stop the request before export; uniform-scale-only fallback is not treated as a successful deformation.

## GLB units and browser try-on

Internal templates, dimensions, deformation thresholds, and mesh quality checks use millimetres. Exported GLBs use metres per the glTF convention. Metadata reports measurements in mm and anchors in m. The viewer measures the loaded model's source width and scales it to the tracked head width, rather than applying a scale factor that assumes every source model has unit width.

## Validation

The CI suite runs `compileall` and `pytest` on Python 3.11. Unit tests validate model contracts, calibration handling, template selection failures, GLB metre conversion, source-mesh immutability, and invalid geometry rejection. Passing unit tests does not replace real-product validation: a production release still requires a labelled holdout of real glasses photos, known frame dimensions, per-image segmentation review, and browser try-on checks.
