"""
Template Registry Module

Provides centralized management and loading of template assets.
"""

from .registry import (
    TEMPLATE_ROOT,
    get_template_dir,
    get_template_assets,
    list_templates,
)

from .loader import TemplateBundle

__all__ = [
    "TEMPLATE_ROOT",
    "get_template_dir",
    "get_template_assets",
    "list_templates",
    "TemplateBundle",
]
