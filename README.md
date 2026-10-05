# Defirmation

Template-based eyewear deformation and virtual try-on pipeline.

## Production-hardening status

This branch adds deployment infrastructure around the existing descriptor/vertex-group deformation engine: asynchronous image/video jobs, quality-gated video keyframes, robust multi-view fusion, structured job artifacts, a browser dashboard, IPD-based Face Mesh try-on, measurement adjustment, Docker/Redis/Celery support, and CI.

**Implemented:** the infrastructure path is wired to the repository's real `DeformationPipeline`. The evaluator reports MAE, median absolute error, p90, maximum error, percent MAE, signed bias, pass rate, archetype aggregates, and worst cases. Accuracy CI is separate, opt-in, and baseline-diffed.

**Unproven:** no real-eyewear ground-truth benchmark has been established. The checked-in three-sample dataset is synthetic plumbing data only. `models/glasses_seg.pt` (or another validated one-class eyewear model) and a real hand-measured dataset are required before accuracy can be claimed.

The remaining validation gate is:

**real photos → real segmentation → accurate measurements → correct template selection → realistic deformation → valid GLB → visual comparison → browser VTO**

Synthetic benchmark results must not be treated as product accuracy.

## Existing deformation engine

The descriptor → vertex groups → rim/bridge/temple/lens deformers → constraint/symmetry solving → smoothing → quality-checking architecture is preserved. The production-hardening work integrates around it rather than replacing it with generic vertex warps.

## Inputs and outputs

- Input: 1–8 images through the job API, with 4–5 recommended, or one short orbit video.
- Video input: quality-gated keyframes are extracted before measurement.
- Output: GLB plus persisted measurements and pipeline metadata.
- Dashboard: job progress, GLB viewer, measurement adjustment, and browser Face Mesh try-on.

## Accuracy data

Canonical ground truth is JSON with `units: "mm"`, required measurement fields, archetype, source, measured_by, and per-field tolerances. A strict CSV importer is provided for bulk hand entry.

Populate `dataset/ground_truth/samples.json` with real measured products and establish `baseline.json` only from a real benchmark run. Do not fabricate baseline values.
