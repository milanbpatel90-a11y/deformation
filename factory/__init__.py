"""Blender Template Factory.

A production-grade pipeline that converts raw GLB eyewear models into
fully-processed templates (descriptor, metadata, thumbnails, GLB) compatible
with the topology-aware deformation engine in ``backend/deformer``.

Run with::

    blender --background --python blender_factory.py -- --input templates/raw

Modules
-------
types
    Shared dataclasses and enums.
utils
    Logging setup, math, and geometry helpers.
importer
    Stage 1: import, clean, normalize scale/origin.
analyzer
    Stage 2: geometric/topological analysis.
component_detector
    Stage 3: classify parts (Frame, Lens, Bridge, Temple...).
splitter
    Stage 4: split meshes when needed, rename parts.
vertex_groups
    Stage 5: build topological vertex groups.
descriptor_generator
    Stage 6: produce descriptor.json for the deformation engine.
metadata_generator
    Stage 7: produce metadata.json.
thumbnail_renderer
    Stage 8: render front / side / 45° thumbnails.
quality_validator
    Stage 9: score and validate the processed template.
registry_builder
    Stage 10: maintain registry.json.
blender_factory
    Stage 11: batch orchestrator (CLI entry point).
"""

from __future__ import annotations

__all__ = [
    "types",
    "utils",
    "importer",
    "analyzer",
    "component_detector",
    "splitter",
    "vertex_groups",
    "descriptor_generator",
    "metadata_generator",
    "thumbnail_renderer",
    "quality_validator",
    "registry_builder",
    "blender_factory",
]
