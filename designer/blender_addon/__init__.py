from __future__ import annotations

from .config import DEFAULT_CONFIG

bl_info = {
    "name": DEFAULT_CONFIG.metadata.name,
    "author": DEFAULT_CONFIG.metadata.author,
    "version": DEFAULT_CONFIG.metadata.version,
    "blender": DEFAULT_CONFIG.metadata.blender_min_version,
    "location": "View3D > Sidebar > Defirmation",
    "description": DEFAULT_CONFIG.metadata.description,
    "category": DEFAULT_CONFIG.metadata.category,
}

try:
    from .operators import register as register_operators
    from .ui import register as register_ui
    from .operators import unregister as unregister_operators
    from .ui import unregister as unregister_ui
except ImportError:  # pragma: no cover - Blender runtime only
    register_operators = None
    unregister_operators = None
    register_ui = None
    unregister_ui = None


def register() -> None:
    """Registers the Blender add-on classes when running inside Blender."""
    if register_operators is not None:
        register_operators()
    if register_ui is not None:
        register_ui()


def unregister() -> None:
    """Unregisters the Blender add-on classes when running inside Blender."""
    if unregister_ui is not None:
        unregister_ui()
    if unregister_operators is not None:
        unregister_operators()

