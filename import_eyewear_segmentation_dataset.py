#!/usr/bin/env python3
"""Safely import the AI-prelabelled 1,000-image eyewear dataset into this repo.

Copies only data directories and provenance files; preserves repository code,
README and .gitignore. ZIP .venv and generated experiments are excluded.
Existing data dirs/metadata are refused unless --replace is explicitly supplied.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

DATA_DIRS = ("images", "masks", "labels")
DATA_FILES = ("data.yaml", "manifest.csv", "human_review_queue.csv", "SUMMARY.json", "UPSTREAM_DATASET_CARD.md", "review_montage.jpg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted dataset folder containing images/, masks/, labels/, manifest.csv and data.yaml")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", type=Path, default=Path("dataset/real_eyewear_1000"))
    parser.add_argument("--replace", action="store_true",
                        help="Replace existing dataset data directories and metadata; back up first")
    args = parser.parse_args()
    source, repo = args.source.resolve(), args.repo_root.resolve()
    dest = args.target if args.target.is_absolute() else repo / args.target
    required = [source / name for name in (*DATA_DIRS, "manifest.csv", "data.yaml")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Source dataset incomplete; missing:\n" + "\n".join(missing), file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    existing = [dest / name for name in (*DATA_DIRS, *DATA_FILES) if (dest / name).exists()]
    if existing and not args.replace:
        print("Refusing to overwrite existing dataset data. Back up first, then pass --replace:\n"
              + "\n".join(str(path) for path in existing), file=sys.stderr)
        return 2
    if args.replace:
        for name in DATA_DIRS:
            path = dest / name
            if path.exists():
                shutil.rmtree(path)
        for name in DATA_FILES:
            path = dest / name
            if path.exists():
                path.unlink()
    for name in DATA_DIRS:
        shutil.copytree(source / name, dest / name)
    for name in DATA_FILES:
        shutil.copy2(source / name, dest / name)
    yaml_path = dest / "data.yaml"
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    yaml_path.write_text("\n".join(
        "path: ." if line.lstrip().startswith("path:") else line
        for line in lines
    ) + "\n", encoding="utf-8")
    print(f"Imported dataset data into: {dest}")
    print("Excluded .venv, generated experiments, model weights, and source helper code.")
    print("WARNING: annotations are AI_PRELABEL_NOT_HUMAN_VERIFIED; visually review before training.")
    print("Run validate_eyewear_dataset.py. Keep the test split out of training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
