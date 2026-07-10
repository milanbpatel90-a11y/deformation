from __future__ import annotations

try:
    import bpy
    from bpy.types import Panel
except ImportError:  # pragma: no cover - Blender runtime only
    bpy = None
    Panel = object  # type: ignore[assignment]


class DEFIRMATION_PT_factory(Panel):
    """Simple Blender UI for the template preparation pipeline."""

    bl_label = "Defirmation Factory"
    bl_idname = "DEFIRMATION_PT_factory"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Defirmation"

    def draw(self, context):  # pragma: no cover - Blender runtime only
        layout = self.layout
        layout.prop(context.scene, "defirmation_source_file")
        layout.operator("defirmation.prepare_template", icon="MODIFIER")


CLASSES = [DEFIRMATION_PT_factory]


def register() -> None:  # pragma: no cover - Blender runtime only
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:  # pragma: no cover - Blender runtime only
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
