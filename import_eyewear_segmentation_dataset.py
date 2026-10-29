#!/usr/bin/env python3
"""Safely import the AI-prelabelled 1,000-image eyewear dataset into the repo.

Copies images, masks, YOLO labels and provenance/review files only. Does not copy
the ZIP's Windows .venv or other generated folders. Existing destinations are
refused unless --replace is explicitly supplied.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

KEEP_DIRS = ("images", "masks", "labels")
KEEP_FILES = (
    "data.yaml", "manifest.csv", "human_review_queue.csv", "README.md",
    "SUMMARY.json", "UPSTREAM_DATASET_CARD.md", "HUMAN_REVIEW_TOOL.html",
    "apply_reviewed_masks.py", "review_montage.jpg",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted dataset folder containing images/, masks/, labels/, manifest.csv and data.yaml")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", type=Path, default=Path("dataset/real_eyewear_1000"),
                        help="Destination relative to repo-root unless absolute")
    parser.add_argument("--replace", action="store_true",
                        help="Delete and replace the destination; back it up first")
    args = parser.parse_args()
    source = args.source.resolve()
    repo_root = args.repo_root.resolve()
    destination = args.target if args.target.is_absolute() else repo_root / args.target

    missing = [str(source / name) for name in (*KEEP_DIRS, "manifest.csv", "data.yaml")
               if not (source / name).exists()]
    if missing:
        print("Source dataset incomplete; missing:\n" + "\n".join(missing), file=sys.stderr)
        return 2
    if destination.exists() and not args.replace:
        print(f"Refusing to overwrite existing destination: {destination}\n"
              "Use a new --target or pass --replace only after backing up.", file=sys.stderr)
        return 2
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()

    for name in KEEP_DIRS:
        shutil.copytree(source / name, destination / name)
    for name in KEEP_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)

    # Make the data config portable and ensure train/val/test references use this root.
    yaml_path = destination / "data.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8")
    yaml_lines = ["path: ." if line.lstrip().startswith("path:") else line
                  for line in yaml_text.splitlines()]
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Imported dataset workspace: {destination}")
    print("WARNING: all source annotations are AI_PRELABEL_NOT_HUMAN_VERIFIED.")
    print("Run validate_eyewear_dataset.py, review masks visually, and keep test images out of training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
