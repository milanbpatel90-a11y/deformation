#!/usr/bin/env python3
"""Validate railpack.json against the schema Railpack actually expects.

This exists because railpack.json silently accepted a wrong shape for
weeks: `packages` was set to a list of apt library names, which Railpack
rejects at deploy time with
`json: cannot unmarshal array into Go struct field Config.packages of
type map[string]string`. System apt libraries belong under
`deploy.aptPackages` (a list of strings); `packages` is reserved for a
map of mise tool versions (e.g. {"python": "3.12"}).

Run directly: python3 scripts/validate_railpack.py [path]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(data: dict) -> list[str]:
    errors: list[str] = []

    if "packages" in data:
        packages = data["packages"]
        if not isinstance(packages, dict):
            errors.append(
                "`packages` must be a map of tool versions (e.g. "
                '{"python": "3.12"}), not a list. System apt libraries '
                "belong under `deploy.aptPackages` instead."
            )
        else:
            for key, value in packages.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    errors.append(
                        f"`packages.{key!r}` must map a string tool name to a "
                        "string version."
                    )

    deploy = data.get("deploy")
    if deploy is not None:
        if not isinstance(deploy, dict):
            errors.append("`deploy` must be an object.")
        else:
            apt_packages = deploy.get("aptPackages")
            if apt_packages is not None and not (
                isinstance(apt_packages, list)
                and all(isinstance(p, str) for p in apt_packages)
            ):
                errors.append("`deploy.aptPackages` must be a list of strings.")

    return errors


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "railpack.json")
    if not path.exists():
        print(f"No {path} found — nothing to validate.")
        return 0

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"{path} is not valid JSON: {exc}")
        return 1

    if not isinstance(data, dict):
        print(f"{path} must contain a JSON object at the top level.")
        return 1

    errors = validate(data)
    if errors:
        print(f"{path} failed validation:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"{path} looks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
