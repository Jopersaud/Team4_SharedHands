"""
extract_video_landmarks.py
--------------------------
Extracts MediaPipe hand landmarks from ASL video datasets.
Handles common folder structures automatically:
  - root/A/video.mp4
  - root/train/A/video.mp4
  - root/A_001.mp4  (flat with label prefix)

Output: artifacts/video_landmarks/{LETTER}/{video_stem}.npy
Each .npy is shape (30, 63) — 30 frames, 21 landmarks × 3 (x,y,z)

Usage:
  python extract_video_landmarks.py --data_dir /path/to/dataset
  python extract_video_landmarks.py --data_dir /path/to/dataset --workers 16
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

# Must be inside __main__ guard on Windows to avoid fork issues
SEQ_LEN = 30
NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3  # 63
VALID_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV"}


def extract_sequence(video_path: str, seq_len: int = SEQ_LEN):
    """
    Sample seq_len evenly-spaced frames from a video and extract landmarks.
    Returns np.ndarray of shape (seq_len, 63) or None on failure.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) < seq_len // 2:
        return None

    mp_hands = mp.solutions.hands
    sequence = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    ) as hands:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                flat = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            else:
                flat = np.zeros(FEATURES, dtype=np.float32)
            sequence.append(flat)

    # Pad to seq_len if some frames failed
    while len(sequence) < seq_len:
        sequence.append(np.zeros(FEATURES, dtype=np.float32))

    return np.array(sequence[:seq_len], dtype=np.float32)


def worker_fn(args):
    video_path, label, out_dir = args
    try:
        out_file = Path(out_dir) / label / f"{Path(video_path).stem}.npy"
        if out_file.exists():
            return "skip"

        seq = extract_sequence(video_path)
        if seq is None:
            return "fail"

        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_file), seq)
        return "ok"
    except Exception as e:
        return f"error:{e}"


def infer_label(path: Path) -> str | None:
    """Try to infer the ASL letter label from folder/filename."""
    # Check parent folders (closest first)
    for part in reversed(path.parts[:-1]):
        upper = part.upper().strip()
        # Direct single letter folder: "A", "B" etc.
        if upper in VALID_LABELS:
            return upper
        # Folder like "letter_a", "class_B", "asl_Z"
        stripped = upper.replace("_", "").replace("-", "").replace("LETTER", "").replace("CLASS", "").replace("ASL", "").strip()
        if stripped in VALID_LABELS:
            return stripped

    # Try filename prefix: "A_001.mp4", "letter_B_002.mp4"
    stem = path.stem.upper()
    for part in stem.split("_"):
        clean = part.strip()
        if clean in VALID_LABELS:
            return clean

    return None


def scan_dataset(data_dir: str):
    """Walk directory tree, return list of (video_path_str, label)."""
    root = Path(data_dir)
    pairs = []
    skipped_no_label = 0
    skipped_ext = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in VIDEO_EXTS:
            skipped_ext += 1
            continue
        label = infer_label(path)
        if label is None:
            skipped_no_label += 1
            continue
        pairs.append((str(path), label))

    if skipped_no_label:
        print(f"  [warn] {skipped_no_label} videos skipped (could not infer A-Z label)")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from ASL video dataset")
    parser.add_argument("--data_dir", required=True, help="Root folder of unzipped video dataset")
    parser.add_argument("--out_dir", default="artifacts/video_landmarks", help="Output directory for .npy sequences")
    parser.add_argument("--workers", type=int, default=min(12, cpu_count()), help="Parallel worker processes")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Frames per sequence (default 30)")
    args = parser.parse_args()

    print(f"\nScanning: {args.data_dir}")
    pairs = scan_dataset(args.data_dir)

    if not pairs:
        print("No labeled videos found. Check --data_dir and folder structure.")
        print("Expected structure: root/LETTER/video.mp4  or  root/train/LETTER/video.mp4")
        sys.exit(1)

    counts = Counter(label for _, label in pairs)
    print(f"\nFound {len(pairs)} labeled videos across {len(counts)} classes:")
    for letter in sorted(counts):
        print(f"  {letter}: {counts[letter]}")

    # Filter out already-done
    out_dir = Path(args.out_dir)
    tasks = [
        (vp, lbl, str(out_dir))
        for vp, lbl in pairs
        if not (out_dir / lbl / f"{Path(vp).stem}.npy").exists()
    ]
    already_done = len(pairs) - len(tasks)
    print(f"\n{already_done} already extracted, {len(tasks)} remaining")

    if not tasks:
        print("Nothing to do — all videos already extracted.")
        return

    print(f"Starting extraction with {args.workers} workers...\n")

    from tqdm import tqdm

    ok = fail = skip = 0
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_fn, tasks), total=len(tasks), unit="video"):
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                fail += 1

    print(f"\nDone: {ok} extracted, {skip} skipped, {fail} failed")
    print(f"Landmarks saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
