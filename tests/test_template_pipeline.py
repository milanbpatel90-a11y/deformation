"""Per-template checks (every template with a descriptor + GLB) and end-to-end deformation.

Complements test_template_parts.py (which covers geometric_metal and the aliasing/quality-gate
behaviour) so that rectangle_plastic and any future template are covered automatically.

A template whose logical parts (Frame, Bridge, LeftRim...) are the SAME mesh object makes
every part-level deformer re-deform the whole frame, yet still "passes" a loose quality
check. These tests make that failure impossible to miss.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline
from backend.pipeline.deformation_pipeline import DeformationQualityError, TemplateNotPreparedError
from backend.template_library.loader import TemplateLibrary

TEMPLATES = Path("templates")
REQUIRED_PARTS = ("Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens", "LeftTemple", "RightTemple")
# Templates that have a descriptor AND a GLB are expected to be fully usable.
USABLE = sorted(p.stem for p in (TEMPLATES / "descriptors").glob("*.json") if (TEMPLATES / f"{p.stem}.glb").exists())


def _measurements(**overrides) -> Measurements:
    base = dict(
        frame_width=138.0, lens_width=52.0, lens_height=42.0, bridge_width=18.0,
        temple_length=140.0, rim_thickness=1.2, material=FrameMaterial.METAL,
        shape=FrameShape.RECTANGLE, nose_pads=False,
    )
    base.update(overrides)
    return Measurements(**base)


@pytest.fixture(scope="module")
def pipeline() -> DeformationPipeline:
    return DeformationPipeline()


def test_there_are_templates_to_check() -> None:
    assert {"geometric_metal", "rectangle_plastic"} <= set(USABLE)


@pytest.mark.parametrize("name", USABLE)
def test_template_is_split_into_distinct_named_parts(name: str, pipeline: DeformationPipeline) -> None:
    info = TemplateLibrary(TEMPLATES).load(name)
    scene, ctx = pipeline._build_context(info, name, _measurements())   # raises if aliased/ill-named
    ctx.assert_distinct_parts()
    for part in REQUIRED_PARTS:
        mesh = ctx.mesh(part)
        assert len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"{name}.{part} is empty"
    ids = [id(ctx.mesh(p)) for p in REQUIRED_PARTS]
    assert len(set(ids)) == len(REQUIRED_PARTS), f"{name}: parts share a mesh object"
    # No part may be most of the whole front: that is the signature of an unsplit mesh.
    front_w = max(ctx.mesh(p).bounds[1][0] for p in ("Frame", "LeftRim", "RightRim")) - min(
        ctx.mesh(p).bounds[0][0] for p in ("Frame", "LeftRim", "RightRim")
    )
    for rim in ("LeftRim", "RightRim"):
        assert ctx.mesh(rim).extents[0] / front_w < 0.62


@pytest.mark.parametrize("name", USABLE)
def test_end_to_end_deformation_passes_quality_and_hits_targets(
    name: str, pipeline: DeformationPipeline, tmp_path: Path
) -> None:
    # A realistic request FOR THIS TEMPLATE: 52x42 mm lens, but never bigger than the template's own
    # lens (a cat-eye front cannot honestly be asked for a 42 mm tall lens; the engine rightly refuses).
    scene = trimesh.load(TEMPLATES / f"{name}.glb", force="scene")
    lens_ext = scene.geometry["LeftLens"].extents
    m = _measurements(
        frame_width=140.0, lens_width=min(52.0, float(lens_ext[0])), lens_height=min(42.0, float(lens_ext[1])),
        bridge_width=18.0, temple_length=140.0,
    )
    out = tmp_path / f"{name}.glb"
    result = pipeline.run_from_measurements(m, out, template_name=name)
    quality = result["quality"]
    assert quality["passed"] and not quality["critical"], quality
    scene = trimesh.load(out, force="scene")
    meshes = {n: scene.geometry[g] for n in scene.graph.nodes_geometry for g in [scene.graph[n][1]]}
    assert all(trimesh.util.is_shape(mm.vertices, (-1, 3)) and mm.vertices.size for mm in meshes.values())
    assert all(abs(mm.vertices).max() < 1000 for mm in meshes.values()), "vertices blew up"
    bridge_w = meshes["Bridge"].extents[0]
    assert 8.0 <= bridge_w <= 40.0
    front = [meshes[p] for p in ("Frame", "LeftRim", "RightRim", "Bridge")]
    front_w = max(f.bounds[1][0] for f in front) - min(f.bounds[0][0] for f in front)
    assert abs(front_w - m.frame_width) / m.frame_width < 0.15


def test_pipeline_quality_error_blocks_export(pipeline: DeformationPipeline) -> None:
    class _Q:
        critical = ["boom"]

    with pytest.raises(DeformationQualityError):
        pipeline._enforce_quality(_Q(), "t")
