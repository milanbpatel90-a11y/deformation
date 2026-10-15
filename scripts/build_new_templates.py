#!/usr/bin/env python3
"""Rebuild the five templates derived from the raw GLBs in ``templates/raw/``.

    raw/cat_eye_glasses.glb -> cat_eye_acetate        (split_cat_eye_acetate.py)
    raw/glasses (5).glb      -> rectangle_acetate      (split_rectangle_acetate.py)
    raw/glasses (3).glb      -> round_metal            (split_round_metal.py)
    raw/glasses (12).glb      -> aviator_metal          (split_aviator_metal.py)
    raw/glasses (1).glb      -> browline_semirimless   (split_semirimless_metal.py)

Each run rewrites ``<name>.glb``, ``<name>.json``, ``descriptors/<name>.json`` and the
``registry.json`` entry. Usage: python scripts/build_new_templates.py [name ...]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

BUILDERS = {
    "cat_eye_acetate": ("split_cat_eye_acetate", "cat_eye_glasses.glb"),
    "rectangle_acetate": ("split_rectangle_acetate", "glasses (5).glb"),
    "round_metal": ("split_round_metal", "glasses (3).glb"),
    "aviator_metal": ("split_aviator_metal", "glasses (12).glb"),
    "browline_semirimless": ("split_semirimless_metal", "glasses (1).glb"),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(BUILDERS)
    for name in wanted:
        module, raw = BUILDERS[name]
        print(f"=== {name} <- raw/{raw}")
        importlib.import_module(module).build(ROOT / "templates" / "raw" / raw, ROOT / "templates")
