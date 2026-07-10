"""Stage 11 — Batch orchestrator (CLI entry point).

Runs the full factory pipeline on every ``*.glb`` in ``templates/raw/``.

Usage (inside Blender):

    blender --background --python factory/blender_factory.py -- --input templates/raw

Or from Blender's Python console:

    import factory.blender_factory as f
    f.main()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
    _BLENDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment]
    _BLENDER_AVAILABLE = False

from .importer import run_stage1, ImportResult
from .analyzer import run_stage2, to_dict as analysis_to_dict
from .component_detector import run_stage3, ComponentClassification
from .splitter import run_stage4, ensure_only_mesh_objects
from .vertex_groups import run_stage5
from .descriptor_generator import run_stage6
from .metadata_generator import run_stage7
from .thumbnail_renderer import run_stage8
from .quality_validator import run_stage9
from .registry_builder import run_stage10
from .types import FactoryPaths, QualityReport
from .utils import configure_logging, get_logger, iter_glb_files, slugify

LOG = get_logger()


# ---------------------------------------------------------------------------
# Pipeline state (per-template)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineContext:
    template_id: str
    template_name: str
    raw_glb: Path
    paths: FactoryPaths

    # Stage outputs
    import_result: ImportResult | None = None
    analysis: Any | None = None
    classification: ComponentClassification | None = None
    descriptor_result: Any | None = None
    metadata_result: Any | None = None
    thumbnail_result: Any | None = None
    quality_report: QualityReport | None = None
    processed_glb_path: Path | None = None

    # Tracking
    started_at: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Single-template pipeline
# ---------------------------------------------------------------------------

def process_one(ctx: PipelineContext) -> tuple[bool, str | None]:
    """Run the full pipeline for one template. Returns (passed, error_message)."""
    ctx.started_at = datetime.now(timezone.utc).isoformat()
    log = get_logger()

    try:
        log.info("=" * 60)
        log.info("Processing %s (%s)", ctx.template_name, ctx.template_id)
        log.info("=" * 60)

        # ---- Stage 1: Import & Normalise ----
        ctx.import_result = run_stage1(ctx.raw_glb)

        # ---- Stage 2: Analyse ----
        ctx.analysis = run_stage2()

        # ---- Stage 3: Detect Components ----
        ctx.classification = run_stage3(ctx.analysis.bbox_mm)

        # ---- Stage 4: Split & Rename ----
        run_stage4(ctx.classification)
        ensure_only_mesh_objects()

        # ---- Stage 5: Vertex Groups ----
        run_stage5(ctx.classification)

        # ---- Save processed GLB (pre-thumbnail) ----
        processed_glb = ctx.paths.processed_dir / f"{ctx.template_id}.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(processed_glb),
            export_format="GLB",
            export_apply=True,
            export_yup=True,
            export_materials="PLACEHOLDER",
            export_image_format="AUTO",
        )
        ctx.processed_glb_path = processed_glb
        log.info("Saved processed GLB → %s", processed_glb)

        # ---- Stage 6: Descriptor ----
        ctx.descriptor_result = run_stage6(
            ctx.template_id,
            ctx.classification,
            ctx.analysis.bbox_mm,
            ctx.paths.descriptors_dir,
        )

        # ---- Stage 7: Metadata (needs quality score later, pass placeholder) ----
        # We'll re-run metadata after quality validation, but for now store placeholder.
        # Actually we need quality score for metadata. Let's defer metadata generation.

        # ---- Stage 8: Thumbnails ----
        ctx.thumbnail_result = run_stage8(ctx.template_id, ctx.paths.thumbnails_dir)

        # ---- Stage 9: Quality Validation ----
        desc_path = ctx.paths.descriptors_dir / f"{ctx.template_id}.json"
        quality_res = run_stage9(
            ctx.template_id,
            ctx.classification,
            ctx.analysis,
            desc_path,
        )
        ctx.quality_report = quality_res.report

        # ---- Stage 7 (re-run): Metadata with actual quality score ----
        ctx.metadata_result = run_stage7(
            ctx.template_id,
            ctx.template_name,
            ctx.classification,
            ctx.analysis.bbox_mm,
            ctx.quality_report.score,
            ctx.paths.metadata_dir,
        )

        # ---- Stage 10: Registry ----
        run_stage10(
            template_id=ctx.template_id,
            template_name=ctx.template_name,
            metadata=ctx.metadata_result.payload,
            quality=ctx.quality_report.score,
            passed=ctx.quality_report.passed,
            descriptor_path=ctx.descriptor_result.output_path,
            metadata_path=ctx.metadata_result.output_path,
            thumbnail_path=ctx.thumbnail_result.front,
            processed_glb_path=ctx.processed_glb_path,
            templates_root=ctx.paths.root,
        )

        log.info("✓ %s — score=%d passed=%s", ctx.template_id, ctx.quality_report.score, ctx.quality_report.passed)
        return ctx.quality_report.passed, None

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        log.error("✗ %s failed: %s", ctx.template_id, err_msg)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return False, err_msg


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blender Template Factory")
    p.add_argument(
        "--input",
        default="templates/raw",
        help="Directory containing raw GLB files (default: templates/raw)",
    )
    p.add_argument(
        "--root",
        default="templates",
        help="Factory root directory (default: templates)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0 = all)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip templates that already have a processed GLB",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Blender passes its own args before "--"; strip them.
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code (0 = success, 1 = any failure)."""
    args = parse_args(sys.argv if argv is None else argv)

    # Configure logging.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_path = Path(args.root) / "factory.log"
    configure_logging(log_path, log_level)

    LOG.info("Blender Template Factory starting...")
    LOG.info("Input: %s", args.input)
    LOG.info("Root: %s", args.root)

    # Resolve paths.
    paths = FactoryPaths.from_root(Path(args.root))
    paths.ensure()

    raw_dir = Path(args.input)
    if not raw_dir.is_absolute():
        raw_dir = paths.root / raw_dir

    glb_files = list(iter_glb_files(raw_dir))
    if not glb_files:
        LOG.warning("No GLB files found in %s", raw_dir)
        return 0

    if args.limit > 0:
        glb_files = glb_files[: args.limit]

    LOG.info("Found %d GLB file(s) to process.", len(glb_files))

    success = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    for idx, glb_path in enumerate(glb_files, 1):
        template_name = glb_path.stem
        template_id = slugify(template_name)

        # Skip check.
        if args.skip_existing:
            processed = paths.processed_dir / f"{template_id}.glb"
            if processed.exists():
                LOG.info("[%d/%d] Skipping %s (already processed)", idx, len(glb_files), template_id)
                continue

        LOG.info("[%d/%d] %s", idx, len(glb_files), template_name)

        ctx = PipelineContext(
            template_id=template_id,
            template_name=template_name,
            raw_glb=glb_path,
            paths=paths,
        )

        passed, err = process_one(ctx)
        if passed:
            success += 1
        else:
            failed += 1
            errors.append((template_id, err or "unknown error"))

    # Summary
    LOG.info("=" * 60)
    LOG.info("BATCH COMPLETE")
    LOG.info("  Total:  %d", len(glb_files))
    LOG.info("  Passed: %d", success)
    LOG.info("  Failed: %d", failed)
    if errors:
        LOG.info("  Errors:")
        for tid, msg in errors:
            LOG.info("    %s: %s", tid, msg)
    LOG.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
