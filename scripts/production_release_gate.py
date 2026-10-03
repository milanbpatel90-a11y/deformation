"""Release gate for a production deformation deployment.

This gate intentionally checks the delivery contract rather than relying on
unit tests alone: real binary template assets, template readiness, the complete
test suite, and the exact-intersection dependency must all be present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from scripts.template_asset_manifest import verify_manifest
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness


def main() -> int:
    os.environ.setdefault("DEFIRM_PRODUCTION_MODE", "1")

    try:
        manifest = verify_manifest()
        readiness = template_readiness(TemplateLibrary())
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "stage": "asset_readiness", "error": str(exc)}, indent=2))
        return 2

    if not readiness["ready"]:
        print(json.dumps({
            "status": "FAIL",
            "stage": "template_readiness",
            "readiness": readiness,
        }, indent=2))
        return 2

    try:
        import open3d  # noqa: F401
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "stage": "collision_validation_dependency",
            "error": f"Open3D is required for production self-intersection checks: {exc}",
        }, indent=2))
        return 2

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        env={**os.environ, "DEFIRM_PRODUCTION_MODE": "1"},
    )
    if result.returncode != 0:
        return result.returncode

    print(json.dumps({
        "status": "PASS",
        "production_mode": True,
        "asset_manifest": manifest,
        "required_template": readiness["required_template"],
        "ready_templates": readiness["templates"],
        "checks": [
            "binary_template_manifest",
            "template_readiness",
            "open3d_exact_intersection_dependency",
            "full_pytest_suite",
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
