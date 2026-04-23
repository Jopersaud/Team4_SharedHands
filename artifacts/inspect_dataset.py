"""
inspect_dataset.py
------------------
Run this FIRST on the other machine to preview the folder structure
before committing to full extraction.

Usage:
  python inspect_dataset.py --data_dir /path/to/unzipped/dataset
  python inspect_dataset.py --data_dir /path/to/unzipped/dataset --depth 4
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV"}
VALID_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def tree(root: Path, depth: int, prefix: str = "", current_depth: int = 0, max_files: int = 3):
    if current_depth >= depth:
        return
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    for d in dirs:
        print(f"{prefix}📁 {d.name}/")
        tree(d, depth, prefix + "   ", current_depth + 1, max_files)

    shown = 0
    for f in files:
        if shown >= max_files:
            remaining = len(files) - shown
            print(f"{prefix}   ... and {remaining} more files")
            break
        print(f"{prefix}📄 {f.name}")
        shown += 1


def count_videos(root: Path):
    label_counts = defaultdict(int)
    unlabeled = 0
    total = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS and path.suffix not in VIDEO_EXTS:
            continue
        total += 1

        # Try to find label
        label = None
        for part in reversed(path.parts[:-1]):
            upper = part.upper().strip()
            if upper in VALID_LABELS:
                label = upper
                break
            stripped = upper.replace("_", "").replace("-", "").replace("LETTER", "").replace("CLASS", "").replace("ASL", "").strip()
            if stripped in VALID_LABELS:
                label = stripped
                break

        if label is None:
            # Try filename
            for part in path.stem.upper().split("_"):
                if part.strip() in VALID_LABELS:
                    label = part.strip()
                    break

        if label:
            label_counts[label] += 1
        else:
            unlabeled += 1

    return dict(label_counts), unlabeled, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth", type=int, default=4, help="Folder tree depth to show")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"ERROR: {args.data_dir} does not exist")
        return

    print(f"\n{'='*60}")
    print(f"Dataset root: {root.resolve()}")
    print(f"{'='*60}\n")

    print("Folder structure preview:")
    tree(root, args.depth)

    print(f"\n{'='*60}")
    print("Counting videos (this may take a moment for large datasets)...")

    counts, unlabeled, total = count_videos(root)

    print(f"\nTotal video files found: {total}")
    print(f"Videos with detectable A-Z label: {sum(counts.values())}")
    print(f"Videos with no detectable label: {unlabeled}")

    if counts:
        print(f"\nPer-class counts ({len(counts)} classes found):")
        for letter in sorted(counts):
            bar = "█" * min(40, counts[letter] // max(1, max(counts.values()) // 40))
            print(f"  {letter}: {counts[letter]:6d}  {bar}")

    if unlabeled > 0:
        print(f"\nWARNING: {unlabeled} videos couldn't be assigned a label.")
        print("You may need to adjust the label detection in extract_video_landmarks.py")
        print("or rename folders to single letters (A, B, C, ...)")

    print(f"\n{'='*60}")
    print("If the above looks correct, run:")
    print(f"  python extract_video_landmarks.py --data_dir {args.data_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
