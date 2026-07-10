from __future__ import annotations

from pathlib import Path

from .template_builder import build_template_from_file, ensure_factory_structure

try:
    import bpy
    from bpy.types import Operator
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Operator = object  # type: ignore[assignment]


class DEFIRMATION_OT_prepare_template(Operator):
    """Runs the milestone-one preparation flow for a single source asset."""

    bl_idname = "defirmation.prepare_template"
    bl_label = "Prepare Template"
    bl_options = {"REGISTER", "UNDO"}

    source_file = None

    @classmethod
    def poll(cls, context):  # pragma: no cover - Blender runtime only
        return bpy is not None and context is not None

    def execute(self, context):  # pragma: no cover - Blender runtime only
        workspace_root = Path.cwd()
        ensure_factory_structure(workspace_root)
        if not getattr(context.scene, "defirmation_source_file", ""):
            self.report({"ERROR"}, "Choose a source GLB first.")
            return {"CANCELLED"}
        report = build_template_from_file(
            context.scene.defirmation_source_file,
            workspace_root=workspace_root,
        )
        if report.errors:
            self.report({"WARNING"}, "; ".join(report.errors))
        else:
            self.report({"INFO"}, f"Prepared {Path(report.source_file).name}")
        return {"FINISHED"}


CLASSES = [DEFIRMATION_OT_prepare_template]


def register() -> None:  # pragma: no cover - Blender runtime only
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.defirmation_source_file = bpy.props.StringProperty(  # type: ignore[attr-defined]
        name="Source GLB",
        subtype="FILE_PATH",
    )


def unregister() -> None:  # pragma: no cover - Blender runtime only
    if bpy is None:
        return
    if hasattr(bpy.types.Scene, "defirmation_source_file"):
        del bpy.types.Scene.defirmation_source_file
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
