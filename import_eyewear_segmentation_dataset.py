#!/usr/bin/env python3
"""Safely import the AI-prelabelled 1,000-image eyewear dataset into this repo.

Copies images, masks, YOLO labels and provenance/review files only. The ZIP's
Windows .venv and generated experiments are intentionally excluded. Existing
destinations are refused unless --replace is explicitly supplied.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

KEEP_DIRS = ("images", "masks", "labels")
KEEP_FILES = ("data.yaml", "manifest.csv", "human_review_queue.csv", "README.md",
              "SUMMARY.json", "UPSTREAM_DATASET_CARD.md", "HUMAN_REVIEW_TOOL.html",
              "apply_reviewed_masks.py", "review_montage.jpg")

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="Extracted dataset folder containing images/, masks/, labels/, manifest.csv and data.yaml")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--target", type=Path, default=Path("dataset/real_eyewear_1000"))
    p.add_argument("--replace", action="store_true",
                   help="Delete destination first; only use after backing up")
    args = p.parse_args()
    source, repo = args.source.resolve(), args.repo_root.resolve()
    dest = args.target if args.target.is_absolute() else repo / args.target
    missing = [str(source / name) for name in (*KEEP_DIRS, "manifest.csv", "data.yaml")
               if not (source / name).exists()]
    if missing:
        print("Source incomplete; missing:\n" + "\n".join(missing), file=sys.stderr)
        return 2
    if dest.exists() and not args.replace:
        print(f"Refusing to overwrite {dest}. Choose another --target or explicitly use --replace after a backup.",
              file=sys.stderr)
        return 2
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    for name in KEEP_DIRS:
        shutil.copytree(source / name, dest / name)
    for name in KEEP_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    yaml_path = dest / "data.yaml"
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    yaml_path.write_text("\n".join("path: ." if line.lstrip().startswith("path:") else line
                                    for line in lines) + "\n", encoding="utf-8")
    print(f"Imported dataset workspace: {dest}")
    print("WARNING: annotations are AI_PRELABEL_NOT_HUMAN_VERIFIED; visually review before training.")
    print("Run validate_eyewear_dataset.py. Keep test split out of training.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
