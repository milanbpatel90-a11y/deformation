"""Benchmark the orbit-video pipeline against *real, annotated* orbit clips.

What this harness is
--------------------
It scores :class:`backend.pipeline.video_pipeline.VideoDeformationPipeline`
against hand-measured ground truth. For every ``<name>.mp4`` (or ``.mov``,
``.m4v``, ``.avi``, ``.mkv``, ``.webm``) in a dataset directory it expects a
sibling ``<name>.json`` annotation, runs the pipeline at 5, 8 and 10 views, and
compares the result with a single-front-image baseline produced by the existing
still-image entry point (``DeformationPipeline.run_from_images``).

What this harness is **not**
----------------------------
It is not evidence. There are currently **zero real annotated orbit videos in
this repository**, so running it today produces ``status = "not_run"`` and a
non-zero exit code -- never a pass. An empty or absent dataset is reported as
"the benchmark has not been run", because "no measurements were compared" and
"every measurement was correct" must never look alike. Synthesised clips are
*not* acceptable input here; see ``docs/ORBIT_VIDEO_BENCHMARK.md``.

Honesty rules baked into the report
-----------------------------------
* ``status`` is ``"not_run"`` when nothing was scored, ``"fail"`` when a scored
  video or a threshold check failed, and ``"pass"`` only when at least one video
  was scored *and* every enabled check passed.
* A threshold check with no data is ``"not_evaluated"`` -- it never counts as a
  pass. A check can only pass when it has data and that data meets the bar.
* ``checks_evaluated`` records which checks actually had evidence, so a reader
  can tell a five-check pass from a two-check pass.
* ``accuracy_claimed`` is ``True`` only for a fully evaluated pass over enough
  real clips, and ``accuracy_claim_note`` always says what the numbers can and
  cannot support.

Conventions follow ``scripts/benchmark_real_products.py``: same CLI style, same
JSON report shape, same markdown + console summary, same refusal to call an
empty fixture a validation run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger("benchmark_orbit_videos")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/benchmark_orbit_videos.py`
    sys.path.insert(0, str(PROJECT_ROOT))

#: Annotation dimensions that carry millimetres. ``rim_thickness`` is optional
#: per-dimension just like the others, but it lives in the same list because it
#: is scored identically.
DIMENSIONS = (
    "frame_width",
    "lens_width",
    "lens_height",
    "bridge_width",
    "temple_length",
    "rim_thickness",
)

#: The only two annotation fields a scored comparison truly needs. Everything
#: else in the schema is optional and is documented as such.
REQUIRED_FIELDS = ("front_frame", "measurements")

OPTIONAL_FIELDS = (
    "product_id",
    "frame_count",
    "fps",
    "side_frame",
    "top_frame",
    "usable_frames",
    "expected_selected_views",
    "category",
    "lighting",
    "background",
)

#: Frame-shape/material families the 30-clip target must span.
CATEGORIES = (
    "metal",
    "acetate",
    "rimless",
    "thick",
    "thin",
    "rectangular",
    "round",
    "cat_eye",
    "wayfarer",
    "aviator",
)

#: View labels the orbit classifier can emit. ``rear`` is listed for completeness
#: even though the current classifier never emits it (see ``ViewClassifier``);
#: adding it here would silently inflate diversity if it ever appeared by mistake.
VIEW_CLASSES = (
    "front",
    "left_front_perspective",
    "right_front_perspective",
    "side",
    "top",
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")

#: Exit codes. 3 exists so that "nothing was compared" can never be mistaken for
#: the 0 that ``benchmark_real_products.py`` uses for a real pass.
EXIT_PASS = 0
EXIT_FAIL = 2
EXIT_NOT_RUN = 3
EXIT_USAGE = 4

#: Defaults for the threshold flags. ``--max-view-class-diversity`` is disabled
#: by default because the classifier's label vocabulary is only five classes
#: wide; with a 5-view target, demanding more than 4 distinct classes is a
#: configuration error rather than a product requirement.
DEFAULTS = {
    "max_mae_mm": 2.0,
    "max_rmse_mm": 3.0,
    "max_error_mm": 5.0,
    "max_template_mismatches": 0,
    "min_confidence_level": "medium",
    "min_selected_views": 5,
    "min_view_class_diversity": None,
    "min_acceptance_pass_rate": 0.9,
}

NOT_RUN_NOTE = (
    "The repository contains zero real annotated orbit videos, so no accuracy claim "
    "is supported. Provide real 360-degree clips with matching annotation JSON in a "
    "dataset directory, then re-run this harness. Synthetic or generated clips do not "
    "count as real product validation."
)


# ── annotation loading ──────────────────────────────────────────────────────
class AnnotationError(ValueError):
    """An annotation file is missing, unreadable or malformed."""


def _is_real_number(value) -> bool:
    """True for a finite int/float, and never for a bool (JSON ``true`` is not a size)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _optional_frame_index(annotation: dict, field: str) -> int | None:
    if field not in annotation or annotation[field] is None:
        return None
    value = annotation[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnnotationError(f"'{field}' must be a non-negative integer frame index")
    return value


def load_annotation(path: Path) -> dict:
    """Load and validate one orbit-video annotation.

    Only ``front_frame`` and ``measurements`` are required, because only they are
    needed for a scored comparison: the front frame locates the baseline image and
    the measurements supply ground truth. Every other documented field is
    optional and is simply absent from the returned context when omitted.

    Raises:
        AnnotationError: the file is missing, is not valid JSON, is not a JSON
            object, lacks a required field, or supplies a non-numeric /
            non-positive measurement.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Annotation file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"Annotation is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise AnnotationError(f"Annotation could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnnotationError("Annotation must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise AnnotationError("Missing required field(s): " + ", ".join(missing))

    front_frame = _optional_frame_index(payload, "front_frame")
    if front_frame is None:
        raise AnnotationError("'front_frame' must be a non-negative integer frame index")

    measurements = payload["measurements"]
    if not isinstance(measurements, dict):
        raise AnnotationError("'measurements' must be an object of millimetre values")

    fields: dict[str, float] = {}
    for dimension in DIMENSIONS:
        if dimension not in measurements or measurements[dimension] is None:
            # Omitted or explicitly null is "the annotator did not supply this".
            # It is never scored, and it is never treated as zero error.
            continue
        value = measurements[dimension]
        if not _is_real_number(value) or float(value) <= 0:
            raise AnnotationError(
                f"'measurements.{dimension}' must be a positive number or null/absent"
            )
        fields[dimension] = float(value)
    if not fields:
        raise AnnotationError(
            "None of the six dimensions in 'measurements' supplied a usable value: "
            + ", ".join(DIMENSIONS)
        )

    category = payload.get("category")
    if category is not None:
        if not isinstance(category, str) or not category.strip():
            raise AnnotationError("'category' must be a non-empty string when present")
        category = category.strip()

    for field in ("frame_count", "expected_selected_views"):
        if payload.get(field) is not None and not _is_real_number(payload[field]):
            raise AnnotationError(f"'{field}' must be a number when present")
    if payload.get("fps") is not None and not _is_real_number(payload["fps"]):
        raise AnnotationError("'fps' must be a number when present")
    usable = payload.get("usable_frames")
    if usable is not None and not (isinstance(usable, list) and all(isinstance(item, int) for item in usable)):
        raise AnnotationError("'usable_frames' must be a list of integers when present")

    supplied = [field for field in OPTIONAL_FIELDS if payload.get(field) is not None]
    return {
        "path": str(path),
        "product_id": payload.get("product_id"),
        "front_frame": front_frame,
        "side_frame": _optional_frame_index(payload, "side_frame"),
        "top_frame": _optional_frame_index(payload, "top_frame"),
        "measurements": fields,
        "category": category,
        "lighting": payload.get("lighting"),
        "background": payload.get("background"),
        "frame_count": payload.get("frame_count"),
        "fps": payload.get("fps"),
        "usable_frames": payload.get("usable_frames"),
        "expected_selected_views": payload.get("expected_selected_views"),
        "unscored_dimensions": [d for d in DIMENSIONS if d not in fields],
        "optional_fields_supplied": supplied,
        "required_fields_present": list(REQUIRED_FIELDS),
    }


def discover_dataset(dataset_dir: Path) -> dict:
    """Pair every video in ``dataset_dir`` with its annotation.

    Returns a dict with ``pairs``, ``unannotated`` (video without a matching JSON)
    and ``orphan_annotations`` (JSON without a matching video). Unpaired files are
    reported rather than silently ignored, because a missing annotation is the
    most likely reason a real clip never gets scored.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return {"pairs": [], "unannotated": [], "orphan_annotations": [], "missing_directory": True}

    videos = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    # ``*.schema.json`` is documentation, not data: the committed template lives
    # in annotations/ today, but a copy dropped in the dataset root must never be
    # mistaken for a real annotation of a clip called "<something>.schema".
    annotations = {
        path.stem: path
        for path in dataset_dir.glob("*.json")
        if path.is_file() and not path.name.endswith(".schema.json")
    }

    pairs, unannotated = [], []
    used = set()
    for video in videos:
        annotation = annotations.get(video.stem)
        if annotation is None:
            unannotated.append(str(video))
            continue
        used.add(video.stem)
        pairs.append({"name": video.stem, "video": video, "annotation": annotation})

    orphan = [str(path) for stem, path in sorted(annotations.items()) if stem not in used]
    return {
        "pairs": pairs,
        "unannotated": unannotated,
        "orphan_annotations": orphan,
        "missing_directory": False,
    }


# ── metrics ─────────────────────────────────────────────────────────────────
def summarize(errors: list[float]) -> dict:
    """MAE / RMSE / max abs error, in millimetres.

    Deliberately identical in spirit to ``scripts/benchmark_real_products.py``: an
    empty list returns ``count = 0`` and ``None`` metrics, never zeros.
    """
    errors = [float(value) for value in errors]
    if not errors:
        return {"count": 0, "mae_mm": None, "rmse_mm": None, "max_abs_error_mm": None}
    return {
        "count": len(errors),
        "mae_mm": sum(abs(value) for value in errors) / len(errors),
        "rmse_mm": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "max_abs_error_mm": max(abs(value) for value in errors),
    }


def measurement_errors(measured: dict, expected: dict) -> dict:
    """Per-dimension signed/absolute/squared error, only where ground truth exists.

    ``expected`` comes from the annotation, so a dimension the annotator did not
    supply is simply absent from the result rather than scored against zero.
    """
    comparison: dict[str, dict] = {}
    for dimension in DIMENSIONS:
        if dimension not in expected:
            continue
        if dimension not in measured or not _is_real_number(measured[dimension]):
            continue
        observed = float(measured[dimension])
        truth = float(expected[dimension])
        signed = observed - truth
        comparison[dimension] = {
            "measured_mm": observed,
            "expected_mm": truth,
            "signed_error_mm": signed,
            "abs_error_mm": abs(signed),
            "squared_error_mm2": signed * signed,
        }
    return comparison


def summarize_comparison(comparison: dict) -> dict:
    """Roll per-dimension errors into the per-dimension MAE/RMSE/max table."""
    return {
        dimension: summarize([entry["signed_error_mm"]])
        for dimension, entry in comparison.items()
    }


def view_class_distribution(per_view: list) -> dict:
    counts = Counter(
        str(entry.get("view"))
        for entry in per_view or []
        if isinstance(entry, dict) and entry.get("view")
    )
    return dict(sorted(counts.items()))


def selection_of(video_section: dict) -> dict:
    """Read the selection block from the ``video`` payload, whichever shape it has.

    The video pipeline has returned the selection summary under ``selection`` and
    under ``selection_summary`` in different revisions, and the key has also been
    missing entirely mid-refactor. All three are handled so the harness reports a
    readable number instead of turning a harmless rename into an error -- and
    without inventing a value when none of them is present.
    """
    if not isinstance(video_section, dict):
        return {}
    # ``selection_summary`` is checked first: it is what the current pipeline
    # populates, and it is the value that actually varies with ``target_views``.
    # ``selection`` is only present in older payloads, where the summary is absent.
    for key in ("selection_summary", "selection"):
        candidate = video_section.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def selected_view_count(video_section: dict, per_view: list) -> int:
    """Views actually kept, from the summary if it exists and the views otherwise."""
    summary = selection_of(video_section)
    for key in ("selected_count", "selected"):
        value = summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return len(per_view or [])


def reported_distribution(video_section: dict, per_view: list) -> dict:
    """Prefer the pipeline's own distribution; fall back to counting ``per_view``.

    The fallback is a faithful count of the returned views, not a guess, and the
    report records which source was used so a reader can tell them apart.
    """
    if isinstance(video_section, dict):
        declared = video_section.get("view_distribution")
        if isinstance(declared, dict) and declared:
            return {str(key): int(value) for key, value in sorted(declared.items())}
    return view_class_distribution(per_view)


def diversity_of(distribution: dict, selected: int) -> dict:
    """How many distinct view classes the selection used, and how evenly.

    Normalised entropy makes 5 views split 2/1/1/1 more legible than a raw class
    count, which cannot tell an even spread from a lopsided one.
    """
    counts = [int(value) for value in distribution.values() if int(value) > 0]
    total = sum(counts)
    if total == 0:
        return {"distinct_classes": 0, "normalised_entropy": None, "max_observed": 0}
    shares = [value / total for value in counts]
    entropy = -sum(share * math.log(share) for share in shares)
    ceiling = math.log(len(VIEW_CLASSES))
    return {
        "distinct_classes": len(counts),
        "normalised_entropy": round(entropy / ceiling, 4) if ceiling > 0 else None,
        "max_observed": max(counts),
        "share_of_selected": round(total / selected, 4) if selected else None,
    }


# ── baseline: the still-image pipeline on one front frame ───────────────────
def extract_frame_image(video: Path, frame_index: int):
    """Decode a single frame by index for the single-image baseline.

    Deliberately plain ``cv2.VideoCapture`` random access: the baseline is "what
    the still-image pipeline would have produced from the product's front photo",
    so it must not inherit any of the orbit pipeline's sampling or quality logic.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Frame {frame_index} could not be decoded from {video}")
    return image


def run_front_image_baseline(pipeline, video: Path, frame_index: int, scratch: Path) -> dict:
    """Run the inherited still-image pipeline on the annotated front frame."""
    import cv2

    image = extract_frame_image(video, frame_index)
    scratch.mkdir(parents=True, exist_ok=True)
    frame_path = scratch / "front_frame.png"
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the baseline front frame")
    frame_path.write_bytes(buffer.tobytes())

    result = pipeline.run_from_images(
        frame_path, output_path=scratch / "baseline.glb", automatic_appearance=True
    )
    return {
        "status": "ok",
        "frame_index": int(frame_index),
        "frame_image": str(frame_path),
        "output_glb": result.get("output_glb"),
        "template": result.get("template"),
        "acceptance_status": (result.get("acceptance") or {}).get("status"),
        "measurements": result.get("measurements"),
    }


# ── per-video evaluation ────────────────────────────────────────────────────
def evaluate_views(
    video_name: str,
    video_path: Path,
    annotation: dict,
    pipeline,
    view_counts: tuple,
    target_fps: float,
    output_dir: Path,
    reference_width_mm: float | None,
) -> dict:
    """Run the pipeline at each requested view count and score each run."""
    expected = annotation["measurements"]
    expected_template = annotation.get("category")
    runs = []
    for target in view_counts:
        record = {
            "target_views": int(target),
            "status": "ok",
            "error": None,
            "selected_views": None,
            "view_distribution": {},
            "view_diversity": {"distinct_classes": 0, "normalised_entropy": None, "max_observed": 0},
            "confidence": {},
            "scale": {},
            "acceptance_status": None,
            "acceptance_passed": None,
            "template": None,
            "template_match": None,
            "measurements": {},
            "comparison": {},
            "metrics": {},
            "mae_mm": None,
            "rmse_mm": None,
            "max_error_mm": None,
            "scored_dimensions": 0,
            "selected_views_target": None,
            "view_distribution_source": None,
            "classification_distribution": None,
            "orbit_geometry": None,
            "per_dimension_provenance": None,
            "pipeline_warnings": [],
            "output_glb": None,
            "manifest_json": None,
            "video_manifest_json": None,
            "timings": {},
        }
        try:
            result = pipeline.run_from_video(
                video_path,
                output_path=output_dir / f"{video_name}_v{int(target)}.glb",
                target_views=int(target),
                target_fps=float(target_fps),
                preview_dir=output_dir / f"previews_{video_name}_v{int(target)}",
                reference_width_mm=reference_width_mm,
            )
            video_section = result.get("video") or {}
            selection = selection_of(video_section)
            per_view = video_section.get("per_view") or []
            measured = result.get("measurements") or {}
            comparison = measurement_errors(measured, expected)
            dimension_metrics = summarize_comparison(comparison)
            selected = selected_view_count(video_section, per_view)
            distribution = reported_distribution(video_section, per_view)
            distribution_source = (
                "pipeline.view_distribution"
                if isinstance(video_section.get("view_distribution"), dict) and video_section.get("view_distribution")
                else "counted_from_per_view"
            )
            template = result.get("template")
            all_signed = [entry["signed_error_mm"] for entry in comparison.values()]
            summary = summarize(all_signed)

            record.update(
                {
                    "selected_views": selected,
                    "selected_views_target": selection.get("target_views"),
                    "view_distribution": distribution,
                    "view_distribution_source": distribution_source,
                    "classification_distribution": video_section.get("classification_distribution"),
                    "view_diversity": diversity_of(distribution, selected),
                    "confidence": video_section.get("confidence") or {},
                    "scale": video_section.get("scale") or {},
                    "orbit_geometry": video_section.get("orbit_geometry"),
                    "per_dimension_provenance": video_section.get("per_dimension_provenance"),
                    "pipeline_warnings": video_section.get("warnings") or [],
                    "acceptance_status": (result.get("acceptance") or {}).get("status"),
                    "acceptance_passed": (result.get("acceptance") or {}).get("status") == "PASS",
                    "template": template,
                    "template_match": (
                        None if expected_template is None or template is None
                        else bool(template == expected_template)
                    ),
                    "measurements": measured,
                    "comparison": comparison,
                    "metrics": dimension_metrics,
                    "mae_mm": summary["mae_mm"],
                    "rmse_mm": summary["rmse_mm"],
                    "max_error_mm": summary["max_abs_error_mm"],
                    "scored_dimensions": len(comparison),
                    "output_glb": result.get("output_glb"),
                    "manifest_json": result.get("manifest_json"),
                    "video_manifest_json": result.get("video_manifest_json"),
                    "timings": (video_section.get("timings") or {}).get("total_seconds"),
                }
            )
        except Exception as exc:  # keep going: one bad clip must not hide the rest
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["error_traceback"] = traceback.format_exc()
            LOGGER.warning(
                "orbit run failed: %s target_views=%s\n%s",
                video_name, target, record["error_traceback"],
            )
        runs.append(record)
    return {
        "name": video_name,
        "video": str(video_path),
        "annotation": annotation["path"],
        "product_id": annotation.get("product_id"),
        "category": expected_template,
        "lighting": annotation.get("lighting"),
        "background": annotation.get("background"),
        "annotated_dimensions": sorted(expected),
        "unscored_dimensions": annotation.get("unscored_dimensions", []),
        "runs": runs,
    }


# ── aggregate report ────────────────────────────────────────────────────────
def aggregate(videos: list[dict], view_counts: tuple) -> dict:
    """Per-view-count aggregate metrics plus per-dimension MAE/RMSE/max tables."""
    by_view: dict[str, dict] = {}
    for target in view_counts:
        signed: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
        comparisons = 0
        selected_counts: list[int] = []
        diversities: list[dict] = []
        confidence_levels = Counter()
        confidence_scores: list[float] = []
        acceptance_total = acceptance_passed = 0
        template_compared = template_matched = 0
        errors = 0
        for video in videos:
            run = next((item for item in video["runs"] if item["target_views"] == target), None)
            if run is None or run["status"] != "ok":
                errors += 1
                continue
            for dimension, entry in run["comparison"].items():
                signed[dimension].append(entry["signed_error_mm"])
            if run["comparison"]:
                comparisons += 1
            if run["selected_views"] is not None:
                selected_counts.append(int(run["selected_views"]))
            diversities.append(run["view_diversity"])
            level = (run["confidence"] or {}).get("level")
            if level:
                confidence_levels[str(level)] += 1
            score = (run["confidence"] or {}).get("score")
            if _is_real_number(score):
                confidence_scores.append(float(score))
            if run["acceptance_status"] is not None:
                acceptance_total += 1
                acceptance_passed += 1 if run["acceptance_passed"] else 0
            if run["template_match"] is not None:
                template_compared += 1
                template_matched += 1 if run["template_match"] else 0

        scored = sum(len(values) for values in signed.values())
        all_signed = [value for values in signed.values() for value in values]
        overall = summarize(all_signed)
        by_view[str(int(target))] = {
            "videos_attempted": len(videos),
            "videos_scored": comparisons,
            "videos_errored": errors,
            "measurements_compared": scored,
            "mae_mm": overall["mae_mm"],
            "rmse_mm": overall["rmse_mm"],
            "max_error_mm": overall["max_abs_error_mm"],
            "per_dimension": {dimension: summarize(values) for dimension, values in signed.items()},
            "selected_views": {
                "count": len(selected_counts),
                "mean": (sum(selected_counts) / len(selected_counts)) if selected_counts else None,
                "min": min(selected_counts) if selected_counts else None,
                "max": max(selected_counts) if selected_counts else None,
            },
            "view_class_diversity": {
                "mean_distinct_classes": (
                    sum(item["distinct_classes"] for item in diversities) / len(diversities)
                    if diversities else None
                ),
                "min_distinct_classes": min((item["distinct_classes"] for item in diversities), default=None),
                "mean_normalised_entropy": (
                    sum(item["normalised_entropy"] for item in diversities if item["normalised_entropy"] is not None)
                    / len([item for item in diversities if item["normalised_entropy"] is not None])
                    if any(item["normalised_entropy"] is not None for item in diversities) else None
                ),
            },
            "confidence": {
                "levels": dict(confidence_levels),
                "mean_score": (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else None,
            },
            "acceptance": {
                "compared": acceptance_total,
                "passed": acceptance_passed,
                "pass_rate": (acceptance_passed / acceptance_total) if acceptance_total else None,
            },
            "template": {
                "compared": template_compared,
                "matched": template_matched,
                "mismatches": template_compared - template_matched,
                "accuracy": (template_matched / template_compared) if template_compared else None,
            },
        }
    return by_view


def aggregate_dimensions(videos: list[dict]) -> dict:
    """Per-dimension metrics pooled over every view count and every video."""
    pooled: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    for video in videos:
        for run in video["runs"]:
            if run["status"] != "ok":
                continue
            for dimension, entry in run["comparison"].items():
                pooled[dimension].append(entry["signed_error_mm"])
    return {dimension: summarize(values) for dimension, values in pooled.items()}


def threshold_checks(report: dict, thresholds: dict, view_counts: tuple) -> dict:
    """Evaluate every enabled CLI threshold, distinguishing pass from no-data.

    ``not_evaluated`` is the third state on purpose: a check with nothing to
    measure must never be recorded as a pass.
    """
    checks: dict[str, dict] = {}

    def record(name: str, value, limit, comparison, note: str = "") -> None:
        if value is None:
            checks[name] = {"status": "not_evaluated", "value": None, "limit": limit, "note": note or "no data"}
            return
        passed = comparison(value, limit)
        checks[name] = {
            "status": "pass" if passed else "fail",
            "value": value,
            "limit": limit,
            "note": note,
        }

    by_view = report["aggregate"]["by_view_count"]
    all_signed_mae = []
    all_rmse = []
    all_max = []
    for target in view_counts:
        entry = by_view[str(int(target))]
        if entry["mae_mm"] is not None:
            all_signed_mae.append(entry["mae_mm"])
        if entry["rmse_mm"] is not None:
            all_rmse.append(entry["rmse_mm"])
        if entry["max_error_mm"] is not None:
            all_max.append(entry["max_error_mm"])

    record(
        "measurement_mae_mm",
        max(all_signed_mae) if all_signed_mae else None,
        thresholds["max_mae_mm"],
        lambda value, limit: value <= limit,
        "worst per-view-count pooled MAE across annotated dimensions",
    )
    record(
        "measurement_rmse_mm",
        max(all_rmse) if all_rmse else None,
        thresholds["max_rmse_mm"],
        lambda value, limit: value <= limit,
        "worst per-view-count pooled RMSE",
    )
    record(
        "measurement_max_error_mm",
        max(all_max) if all_max else None,
        thresholds["max_error_mm"],
        lambda value, limit: value <= limit,
        "worst single-dimension single-clip absolute error",
    )

    compared = sum(by_view[str(int(target))]["template"]["compared"] for target in view_counts)
    mismatches = sum(by_view[str(int(target))]["template"]["mismatches"] for target in view_counts)
    checks["template_mismatches"] = (
        {
            "status": "pass" if mismatches <= thresholds["max_template_mismatches"] else "fail",
            "value": mismatches,
            "limit": thresholds["max_template_mismatches"],
            "compared": compared,
            "note": "annotations without a 'category' cannot be compared and are not counted",
        }
        if compared
        else {
            "status": "not_evaluated",
            "value": None,
            "limit": thresholds["max_template_mismatches"],
            "compared": 0,
            "note": "no annotation supplied 'category'; template accuracy is not evaluated",
        }
    )

    order = {"low": 0, "medium": 1, "high": 2}
    floor = thresholds["min_confidence_level"]
    worst = None
    for target in view_counts:
        for level in by_view[str(int(target))]["confidence"]["levels"]:
            if level in order and (worst is None or order[level] < order[worst]):
                worst = level
    record(
        "confidence_level",
        worst,
        floor,
        lambda value, limit: order.get(value, -1) >= order.get(limit, 0),
        "worst confidence level observed at any view count",
    )

    selected = [
        by_view[str(int(target))]["selected_views"]["min"] for target in view_counts
        if by_view[str(int(target))]["selected_views"]["min"] is not None
    ]
    record(
        "selected_views",
        min(selected) if selected else None,
        thresholds["min_selected_views"],
        lambda value, limit: value >= limit,
        "fewest views actually selected at any view count",
    )

    diversity = thresholds.get("min_view_class_diversity")
    if diversity is None:
        checks["view_class_diversity"] = {
            "status": "disabled",
            "value": None,
            "limit": None,
            "note": "disabled by default; pass --min-view-class-diversity to enable",
        }
    else:
        observed = [
            by_view[str(int(target))]["view_class_diversity"]["min_distinct_classes"]
            for target in view_counts
            if by_view[str(int(target))]["view_class_diversity"]["min_distinct_classes"] is not None
        ]
        record(
            "view_class_diversity",
            min(observed) if observed else None,
            diversity,
            lambda value, limit: value >= limit,
            "fewest distinct view classes in any single clip",
        )

    accepted = sum(by_view[str(int(target))]["acceptance"]["compared"] for target in view_counts)
    passed = sum(by_view[str(int(target))]["acceptance"]["passed"] for target in view_counts)
    record(
        "acceptance_pass_rate",
        (passed / accepted) if accepted else None,
        thresholds["min_acceptance_pass_rate"],
        lambda value, limit: value >= limit,
        "fraction of runs whose exported GLB reported acceptance PASS",
    )
    return checks


def build_report(
    dataset_dir: Path,
    output_dir: Path,
    videos: list[dict],
    discovery: dict,
    view_counts: tuple,
    thresholds: dict,
    baseline: dict,
    skipped_baseline_clips: list[dict],
) -> dict:
    """Assemble the machine-readable report, including its honesty metadata."""
    attempted = len(videos)
    scored_runs = sum(
        1 for video in videos for run in video["runs"] if run["status"] == "ok" and run["comparison"]
    )
    errored_runs = sum(1 for video in videos for run in video["runs"] if run["status"] != "ok")

    report = {
        "schema_version": 1,
        "harness": "scripts/benchmark_orbit_videos.py",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "view_counts": [int(value) for value in view_counts],
        "thresholds": dict(thresholds),
        "dataset": {
            "videos_attempted": attempted,
            "annotations_loaded": sum(1 for video in videos if video.get("annotation")),
            "videos_without_annotation": discovery.get("unannotated", []),
            "annotations_without_video": discovery.get("orphan_annotations", []),
            "dataset_directory_present": not discovery.get("missing_directory", False),
        },
        "baseline": {**baseline, "skipped_clips": skipped_baseline_clips},
        "videos": videos,
        "aggregate": {
            "by_view_count": aggregate(videos, view_counts),
            "per_dimension_all_views": aggregate_dimensions(videos),
        },
        "totals": {
            "videos_attempted": attempted,
            "runs_scored": scored_runs,
            "runs_errored": errored_runs,
            "measurements_compared": sum(
                len(run["comparison"]) for video in videos for run in video["runs"] if run["status"] == "ok"
            ),
        },
    }
    report["checks"] = threshold_checks(report, thresholds, view_counts)
    report["verdict"] = verdict(report)
    return report


def verdict(report: dict) -> dict:
    """Turn the raw counts into an explicit, honest pass/fail/not-run verdict."""
    totals = report["totals"]
    checks = report["checks"]
    enabled = {name: check for name, check in checks.items() if check["status"] != "disabled"}
    evaluated = {name: check for name, check in enabled.items() if check["status"] != "not_evaluated"}
    failures = [name for name, check in enabled.items() if check["status"] == "fail"]

    if totals["videos_attempted"] == 0 or totals["measurements_compared"] == 0:
        status = "not_run"
        summary = (
            "NOT RUN -- no annotated orbit video produced a scored measurement. "
            "This is not a pass, and no accuracy claim is supported."
        )
    elif failures or totals["runs_errored"]:
        status = "fail"
        summary = (
            "FAIL -- " + ("threshold check(s) failed: " + ", ".join(failures) if failures else "")
            + ("; " if failures and totals["runs_errored"] else "")
            + (f"{totals['runs_errored']} run(s) errored" if totals["runs_errored"] else "")
        ).strip()
    else:
        status = "pass"
        summary = "PASS -- every enabled threshold check was evaluated and met."

    return {
        "status": status,
        "summary": summary,
        "checks_enabled": sorted(enabled),
        "checks_evaluated": sorted(evaluated),
        "checks_not_evaluated": sorted(
            name for name, check in enabled.items() if check["status"] == "not_evaluated"
        ),
        "failures": failures,
        "accuracy_claimed": bool(status == "pass"),
        # True whenever *any* real measurement was compared, which is a weaker and
        # more useful statement than "the benchmark passed".
        "measured_against_real_annotations": bool(totals["measurements_compared"]),
        "accuracy_claim_note": (
            "Zero real annotated orbit videos are present in this repository, so no "
            "accuracy claim is supported by this run. "
            if status == "not_run"
            else "Numbers describe only the real clips scored in this run; they are not a "
            "production accuracy guarantee and say nothing about unseen products or "
            "categories absent from the dataset."
        ),
        "disclaimer": NOT_RUN_NOTE,
    }


# ── markdown summary ────────────────────────────────────────────────────────
def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def render_markdown(report: dict) -> str:
    """Human-readable summary. Every empty table says why it is empty."""
    verdict_info = report["verdict"]
    lines: list[str] = []
    lines.append("# Orbit video benchmark report")
    lines.append("")
    lines.append(f"- **Status: `{verdict_info['status'].upper()}`**")
    lines.append(f"- {verdict_info['summary']}")
    lines.append(f"- Dataset: `{report['dataset_dir']}`")
    lines.append(f"- View counts compared: {', '.join(str(v) for v in report['view_counts'])}")
    lines.append(f"- Videos attempted: {report['totals']['videos_attempted']} "
                 f"(annotations loaded: {report['dataset']['annotations_loaded']})")
    lines.append(f"- Runs scored: {report['totals']['runs_scored']} "
                 f"(errored: {report['totals']['runs_errored']})")
    lines.append(f"- Measurements compared against real ground truth: "
                 f"{report['totals']['measurements_compared']}")
    lines.append("")
    lines.append("> " + verdict_info["accuracy_claim_note"].strip())
    lines.append("")

    lines.append("## Threshold checks")
    lines.append("")
    lines.append("| check | status | observed | limit | note |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, check in report["checks"].items():
        observed = check.get("value")
        observed = f"{observed:.4f}" if isinstance(observed, float) else _fmt(observed)
        lines.append(
            f"| `{name}` | {check['status']} | {observed} | {_fmt(check.get('limit'))} | "
            f"{check.get('note', '')} |"
        )
    lines.append("")
    lines.append(f"Checks evaluated: {', '.join(verdict_info['checks_evaluated']) or 'none'}  ")
    lines.append(f"Checks NOT evaluated (no data): "
                 f"{', '.join(verdict_info['checks_not_evaluated']) or 'none'}")
    lines.append("")

    lines.append("## Per-view-count aggregate")
    lines.append("")
    lines.append("| views | scored | errored | measurements | MAE mm | RMSE mm | max err mm | mean selected | "
                 "mean distinct classes | acceptance pass rate | template accuracy | confidence levels |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for target in report["view_counts"]:
        entry = report["aggregate"]["by_view_count"][str(target)]
        lines.append(
            "| {views} | {scored} | {errored} | {measured} | {mae} | {rmse} | {maxe} | {selected} | "
            "{classes} | {accept} | {template} | {conf} |".format(
                views=target,
                scored=entry["videos_scored"],
                errored=entry["videos_errored"],
                measured=entry["measurements_compared"],
                mae=_fmt(entry["mae_mm"]),
                rmse=_fmt(entry["rmse_mm"]),
                maxe=_fmt(entry["max_error_mm"]),
                selected=_fmt(entry["selected_views"]["mean"]),
                classes=_fmt(entry["view_class_diversity"]["mean_distinct_classes"]),
                accept=_fmt(entry["acceptance"]["pass_rate"], 3),
                template=_fmt(entry["template"]["accuracy"], 3),
                conf=json.dumps(entry["confidence"]["levels"]) if entry["confidence"]["levels"] else "n/a",
            )
        )
    if report["totals"]["measurements_compared"] == 0:
        lines.append("")
        lines.append("_Every metric above is `n/a`: **no annotated clip produced a comparison, so this "
                     "table is empty rather than perfect.**_")
    lines.append("")

    lines.append("## Per-dimension error (pooled over all view counts)")
    lines.append("")
    lines.append("| dimension | compared | MAE mm | RMSE mm | max abs err mm |")
    lines.append("| --- | --- | --- | --- | --- |")
    for dimension, metrics in report["aggregate"]["per_dimension_all_views"].items():
        lines.append(
            f"| {dimension} | {metrics['count']} | {_fmt(metrics['mae_mm'])} | "
            f"{_fmt(metrics['rmse_mm'])} | {_fmt(metrics['max_abs_error_mm'])} |"
        )
    lines.append("")
    lines.append("Dimensions absent from every annotation are reported as `count = 0` / `n/a`. They are "
                 "not zero-error, and no substitution or imputation is performed.")
    lines.append("")

    lines.append("## Single-front-image baseline comparison")
    lines.append("")
    baseline = report["baseline"]
    lines.append(f"- Baseline method: {baseline.get('method', 'n/a')}")
    lines.append(f"- Baseline status: `{baseline.get('status')}`")
    if baseline.get("note"):
        lines.append(f"- Note: {baseline['note']}")
    comparisons = baseline.get("comparisons") or []
    if comparisons:
        lines.append("")
        lines.append("| video | views | video MAE mm | baseline MAE mm | delta (video - baseline) mm |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in comparisons:
            lines.append(
                f"| {row['video']} | {row['target_views']} | {_fmt(row['video_mae_mm'])} | "
                f"{_fmt(row['baseline_mae_mm'])} | {_fmt(row['delta_mae_mm'])} |"
            )
    else:
        lines.append("")
        lines.append("_No baseline comparison exists, so nothing can be said about whether the orbit "
                     "pipeline beats a single front photo._")
    lines.append("")

    lines.append("## Per-video detail")
    lines.append("")
    lines.append("| video | category | views | status | selected | classes | MAE mm | max err mm | template | "
                 "template match | acceptance | confidence |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    if not report["videos"]:
        lines.append("| _none_ | | | | | | | | | | | |")
    for video in report["videos"]:
        for run in video["runs"]:
            lines.append(
                "| {name} | {category} | {views} | {status} | {selected} | {classes} | {mae} | {maxe} | "
                "{template} | {match} | {accept} | {confidence} |".format(
                    name=video["name"],
                    category=video.get("category") or "n/a",
                    views=run["target_views"],
                    status=run["status"],
                    selected=_fmt(run["selected_views"]),
                    classes=run["view_diversity"]["distinct_classes"],
                    mae=_fmt(run["mae_mm"]),
                    maxe=_fmt(run["max_error_mm"]),
                    template=run["template"] or "n/a",
                    match=("n/a" if run["template_match"] is None else str(run["template_match"])),
                    accept=run["acceptance_status"] or "n/a",
                    confidence=(run["confidence"] or {}).get("level") or "n/a",
                )
            )
    lines.append("")

    lines.append("## Dataset bookkeeping")
    lines.append("")
    lines.append(f"- Videos with no matching annotation JSON: "
                 f"{len(report['dataset']['videos_without_annotation'])}")
    for path in report["dataset"]["videos_without_annotation"]:
        lines.append(f"  - `{path}`")
    lines.append(f"- Annotation JSONs with no matching video: "
                 f"{len(report['dataset']['annotations_without_video'])}")
    for path in report["dataset"]["annotations_without_video"]:
        lines.append(f"  - `{path}`")
    benchmark_dir = Path(report["output_dir"])
    lines.append("")
    lines.append(f"Annotation template: `dataset/orbit_benchmark/annotations/EXAMPLE.schema.json`. "
                 f"Method and target: `docs/ORBIT_VIDEO_BENCHMARK.md`.")
    lines.append("")
    lines.append(f"Machine-readable report: `{(benchmark_dir / 'report.json').as_posix()}`")
    return "\n".join(lines)


# ── console summary ─────────────────────────────────────────────────────────
def render_console(report: dict) -> str:
    """Fixed-width console table, in the spirit of the other benchmark scripts."""
    verdict_info = report["verdict"]
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("ORBIT VIDEO BENCHMARK -- real annotated 360-degree clips")
    lines.append("=" * 96)
    lines.append(f"dataset : {report['dataset_dir']}")
    lines.append(f"status  : {verdict_info['status'].upper()}  ({verdict_info['summary']})")
    lines.append("")
    header = (f"{'views':>5} {'scored':>7} {'err':>4} {'meas':>5} {'MAE':>8} {'RMSE':>8} "
              f"{'maxerr':>8} {'sel':>5} {'cls':>4} {'accept':>7} {'tmpl':>6} {'conf':>18}")
    lines.append(header)
    lines.append("-" * len(header))
    for target in report["view_counts"]:
        entry = report["aggregate"]["by_view_count"][str(target)]
        levels = ",".join(f"{k}:{v}" for k, v in entry["confidence"]["levels"].items()) or "n/a"
        lines.append(
            f"{target:>5} {entry['videos_scored']:>7} {entry['videos_errored']:>4} "
            f"{entry['measurements_compared']:>5} {_fmt(entry['mae_mm']):>8} {_fmt(entry['rmse_mm']):>8} "
            f"{_fmt(entry['max_error_mm']):>8} {_fmt(entry['selected_views']['mean']):>5} "
            f"{_fmt(entry['view_class_diversity']['mean_distinct_classes']):>4} "
            f"{_fmt(entry['acceptance']['pass_rate'], 3):>7} {_fmt(entry['template']['accuracy'], 3):>6} "
            f"{levels:>18}"
        )
    lines.append("")
    lines.append("per-video:")
    if not report["videos"]:
        lines.append("  (no annotated videos found)")
    for video in report["videos"]:
        for run in video["runs"]:
            detail = (f"  {video['name'][:24]:<24} v{run['target_views']:<3} {run['status']:<6} "
                      f"sel={_fmt(run['selected_views']):<4} cls={run['view_diversity']['distinct_classes']} "
                      f"MAE={_fmt(run['mae_mm']):<7} max={_fmt(run['max_error_mm']):<7} "
                      f"tmpl={run['template'] or 'n/a':<24} match={run['template_match']} "
                      f"acc={run['acceptance_status'] or 'n/a'}")
            lines.append(detail)
            if run["status"] != "ok":
                lines.append(f"      error: {run['error']}")
    lines.append("")
    lines.append("threshold checks:")
    for name, check in report["checks"].items():
        observed = check.get("value")
        observed = f"{observed:.4f}" if isinstance(observed, float) else _fmt(observed)
        flag = {"pass": "PASS", "fail": "FAIL", "not_evaluated": "no-data", "disabled": "off"}[check["status"]]
        lines.append(f"  {name:<28} {flag:<8} observed={observed:<10} limit={_fmt(check.get('limit'))}")
    lines.append("")
    if verdict_info["status"] == "not_run":
        lines.append("BENCHMARK NOT RUN: no real annotated orbit videos were scored.")
        lines.append("  An empty dataset is reported as NOT RUN, never as zero errors / pass.")
        lines.append("  " + verdict_info["accuracy_claim_note"].strip())
    lines.append("=" * 96)
    return "\n".join(lines)


# ── orchestration ───────────────────────────────────────────────────────────
def _load_pipelines(front_image_baseline: bool):
    """Import the pipelines lazily so `--help` and metric tests stay dependency-free."""
    from backend.pipeline.deformation_pipeline import DeformationPipeline
    from backend.pipeline.video_pipeline import VideoDeformationPipeline

    still = DeformationPipeline()
    orbit = VideoDeformationPipeline.sharing(still) if front_image_baseline else VideoDeformationPipeline()
    return still, orbit


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    view_counts: tuple = (5, 8, 10),
    target_fps: float = 6.0,
    thresholds: dict | None = None,
    front_image_baseline: bool = True,
    reference_width_mm: float | None = None,
    pipeline_factory=None,
) -> dict:
    """Score every annotated clip and return ``(report, exit_code, console_text)``.

    A missing/empty dataset returns early with ``status = "not_run"`` and
    :data:`EXIT_NOT_RUN` *before* any model is loaded, so it is cheap and it can
    never be mistaken for a pass.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    view_counts = tuple(int(value) for value in view_counts)
    thresholds = {**DEFAULTS, **(thresholds or {})}

    discovery = discover_dataset(dataset_dir)
    baseline = {
        "method": "DeformationPipeline.run_from_images on the annotated front frame",
        "status": "not_run",
        "note": (
            "Skipped because no annotated clip was scored." if not discovery["pairs"] else ""
        ),
        "comparisons": [],
    }

    pairs = discovery["pairs"]
    if not pairs:
        missing = discovery.get("missing_directory", False)
        suffixes = "/".join(suffix.lstrip(".") for suffix in VIDEO_SUFFIXES)
        reason = (
            f"dataset directory does not exist: {dataset_dir}"
            if missing
            else f"no <name>.{suffixes} clip with a matching <name>.json annotation was found "
                 f"in {dataset_dir}"
        )
        report = {
            "schema_version": 1,
            "harness": "scripts/benchmark_orbit_videos.py",
            "dataset_dir": str(dataset_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "view_counts": list(view_counts),
            "thresholds": thresholds,
            "dataset": {
                "videos_attempted": 0,
                "annotations_loaded": 0,
                "videos_without_annotation": discovery.get("unannotated", []),
                "annotations_without_video": discovery.get("orphan_annotations", []),
                "dataset_directory_present": not missing,
                "reason_empty": reason,
            },
            "baseline": {**baseline, "skipped_clips": []},
            "videos": [],
            "aggregate": {"by_view_count": aggregate([], view_counts), "per_dimension_all_views": aggregate_dimensions([])},
            "totals": {"videos_attempted": 0, "runs_scored": 0, "runs_errored": 0, "measurements_compared": 0},
        }
        report["checks"] = threshold_checks(report, thresholds, view_counts)
        report["verdict"] = verdict(report)
        return {"report": report, "exit_code": EXIT_NOT_RUN, "console": render_console(report)}

    still_pipeline, orbit_pipeline = (
        pipeline_factory() if pipeline_factory is not None else _load_pipelines(front_image_baseline)
    )

    videos: list[dict] = []
    baseline_comparisons: list[dict] = []
    skipped_baseline: list[dict] = []
    for pair in pairs:
        name, video_path = pair["name"], pair["video"]
        clip_dir = output_dir / name
        try:
            annotation = load_annotation(pair["annotation"])
        except AnnotationError as exc:
            videos.append(
                {
                    "name": name,
                    "video": str(video_path),
                    "annotation": str(pair["annotation"]),
                    "annotation_error": str(exc),
                    "runs": [],
                }
            )
            continue

        entry = evaluate_views(
            video_name=name,
            video_path=video_path,
            annotation=annotation,
            pipeline=orbit_pipeline,
            view_counts=view_counts,
            target_fps=target_fps,
            output_dir=clip_dir,
            reference_width_mm=reference_width_mm,
        )
        videos.append(entry)

        if not front_image_baseline:
            # Documented as skipped, never silently omitted and never faked.
            skipped_baseline.append({"video": name, "error": "skipped: --no-front-image-baseline"})
            continue
        try:
            baseline_result = run_front_image_baseline(
                still_pipeline, video_path, annotation["front_frame"], clip_dir / "baseline"
            )
        except Exception as exc:
            skipped_baseline.append(
                {"video": name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        baseline_errors = measurement_errors(baseline_result.get("measurements") or {}, annotation["measurements"])
        baseline_metrics = summarize([entry["signed_error_mm"] for entry in baseline_errors.values()])
        for run in entry["runs"]:
            if run["status"] != "ok" or run["mae_mm"] is None or baseline_metrics["mae_mm"] is None:
                continue
            baseline_comparisons.append(
                {
                    "video": name,
                    "target_views": run["target_views"],
                    "video_mae_mm": run["mae_mm"],
                    "baseline_mae_mm": baseline_metrics["mae_mm"],
                    "delta_mae_mm": run["mae_mm"] - baseline_metrics["mae_mm"],
                    "video_max_error_mm": run["max_error_mm"],
                    "baseline_max_error_mm": baseline_metrics["max_abs_error_mm"],
                    "baseline_template": baseline_result.get("template"),
                    "video_template": run["template"],
                }
            )

    baseline["comparisons"] = baseline_comparisons
    baseline["skipped_clips"] = skipped_baseline
    # Status is only ever set from what actually happened: a request to skip is
    # recorded as skipped, and a request that produced nothing is recorded as
    # no_data -- never as a silently missing baseline.
    all_skipped_by_flag = bool(skipped_baseline) and all(
        item["error"].startswith("skipped:") for item in skipped_baseline
    )
    if not front_image_baseline or all_skipped_by_flag:
        baseline["status"] = "skipped_by_flag"
        baseline["note"] = "Disabled with --no-front-image-baseline."
    elif baseline_comparisons:
        baseline["status"] = "ok"
    elif skipped_baseline:
        baseline["status"] = "unavailable"
        baseline["note"] = (
            "The still-image baseline could not be produced for any clip; see skipped_clips. "
            "No baseline result is reported rather than a fabricated one."
        )
    else:
        baseline["status"] = "no_data"
        baseline["note"] = "No clip produced both a video and a baseline measurement to compare."

    report = build_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        videos=videos,
        discovery=discovery,
        view_counts=view_counts,
        thresholds=thresholds,
        baseline=baseline,
        skipped_baseline_clips=skipped_baseline,
    )
    status = report["verdict"]["status"]
    exit_code = EXIT_PASS if status == "pass" else EXIT_NOT_RUN if status == "not_run" else EXIT_FAIL
    return {"report": report, "exit_code": exit_code, "console": render_console(report)}


def write_outputs(report: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the orbit-video pipeline against real annotated 360-degree clips. "
            "An empty dataset is reported as NOT RUN, never as a pass."
        )
    )
    parser.add_argument("dataset", type=Path, help="Directory of <name>.mp4 + <name>.json pairs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark/orbit"))
    parser.add_argument(
        "--views", type=int, nargs="+", default=[5, 8, 10],
        help="target_views values to compare (default: 5 8 10)",
    )
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument("--max-mae-mm", type=float, default=DEFAULTS["max_mae_mm"])
    parser.add_argument("--max-rmse-mm", type=float, default=DEFAULTS["max_rmse_mm"])
    parser.add_argument("--max-error-mm", type=float, default=DEFAULTS["max_error_mm"])
    parser.add_argument("--max-template-mismatches", type=int, default=DEFAULTS["max_template_mismatches"])
    parser.add_argument(
        "--min-confidence-level", choices=("low", "medium", "high"),
        default=DEFAULTS["min_confidence_level"],
    )
    parser.add_argument("--min-selected-views", type=int, default=DEFAULTS["min_selected_views"])
    parser.add_argument(
        "--min-view-class-diversity", type=int, default=None,
        help="minimum distinct view classes per clip; disabled unless supplied",
    )
    parser.add_argument("--min-acceptance-pass-rate", type=float, default=DEFAULTS["min_acceptance_pass_rate"])
    parser.add_argument(
        "--no-front-image-baseline", action="store_true",
        help="skip the single-front-image baseline (documented as skipped, not faked)",
    )
    parser.add_argument(
        "--reference-width-mm", type=float, default=None,
        help="rescale measurements against a known real frame width in mm",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    thresholds = {
        "max_mae_mm": args.max_mae_mm,
        "max_rmse_mm": args.max_rmse_mm,
        "max_error_mm": args.max_error_mm,
        "max_template_mismatches": args.max_template_mismatches,
        "min_confidence_level": args.min_confidence_level,
        "min_selected_views": args.min_selected_views,
        "min_view_class_diversity": args.min_view_class_diversity,
        "min_acceptance_pass_rate": args.min_acceptance_pass_rate,
    }
    outcome = run_benchmark(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        view_counts=tuple(args.views),
        target_fps=args.target_fps,
        thresholds=thresholds,
        front_image_baseline=not args.no_front_image_baseline,
        reference_width_mm=args.reference_width_mm,
    )
    report = outcome["report"]
    paths = write_outputs(report, args.output_dir)
    print(outcome["console"])
    print(f"report : {paths['json']}")
    print(f"summary: {paths['markdown']}")
    if report["verdict"]["status"] == "not_run" and report["videos"]:
        print("NOTE: clips were found but none produced a scored comparison; see 'videos' in the report.")
    return outcome["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
