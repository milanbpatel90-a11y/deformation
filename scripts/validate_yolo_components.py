"""Validate YOLO segmentation labels against the canonical component contract."""

from __future__ import annotations

import argparse
from pathlib import Path

COMPONENT_CLASS_COUNT = 9


def validate_label_file(path: Path) -> list[str]:
    errors: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            errors.append(f"{path}:{line_no}: need class + at least 3 points")
            continue
        if (len(parts) - 1) % 2 != 0:
            errors.append(f"{path}:{line_no}: odd number of polygon coordinates")
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{path}:{line_no}: invalid class id {parts[0]!r}")
            continue
        if not 0 <= class_id < COMPONENT_CLASS_COUNT:
            errors.append(
                f"{path}:{line_no}: class id {class_id} outside 0..{COMPONENT_CLASS_COUNT - 1}"
            )
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric coordinate")
            continue
        if any(v < 0.0 or v > 1.0 for v in coords):
            errors.append(f"{path}:{line_no}: coordinates must be normalized to [0,1]")
        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) >= 3:
            area2 = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                area2 += xs[i] * ys[j] - xs[j] * ys[i]
            if abs(area2) < 1e-8:
                errors.append(f"{path}:{line_no}: zero-area polygon")
    return errors


def validate_split(root: Path, split: str) -> tuple[int, list[str]]:
    labels = root / split
    if not labels.exists():
        return 0, [f"missing label directory: {labels}"]
    files = sorted(labels.glob("*.txt"))
    errors: list[str] = []
    for path in files:
        errors.extend(validate_label_file(path))
    return len(files), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/labels_components"),
        help="Root containing train/ and val/ component labels",
    )
    args = parser.parse_args()

    total_files = 0
    all_errors: list[str] = []
    for split in ("train", "val"):
        count, errors = validate_split(args.root, split)
        total_files += count
        all_errors.extend(errors)
        print(f"{split}: {count} label files")

    print(f"total: {total_files} label files")
    if all_errors:
        print("\n".join(all_errors))
        print(f"\nFAILED: {len(all_errors)} issue(s)")
        return 1
    print("PASS: all component labels satisfy the format contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
