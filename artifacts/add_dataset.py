"""
Extracts MediaPipe landmarks from archive(1)/own_dataset, appends to
asl_landmarks.csv, then retrains and converts the model to TF.js.

Run from artifacts/:
    python add_dataset.py
"""
import os, csv, io, zipfile
import cv2
import mediapipe as mp
import numpy as np

BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "archive(5)", "asl-numbers-alphabet-dataset")
CSV_PATH    = os.path.join(BASE, "asl_landmarks.csv")

# Letters the model supports (skip 'space' — not in original 26-class model)
VALID_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

new_rows = []
failed = 0

for letter in sorted(os.listdir(DATASET_DIR)):
    label = letter.upper()
    if label not in VALID_LABELS:
        print(f"Skipping '{letter}' (not in A-Z)")
        continue

    folder = os.path.join(DATASET_DIR, letter)
    if not os.path.isdir(folder):
        continue

    images = os.listdir(folder)
    print(f"Processing {label}: {len(images)} images...", end=" ", flush=True)
    ok = 0

    for img_file in images:
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            row = [coord for point in lm for coord in (point.x, point.y, point.z)]
            row.append(label)
            new_rows.append(row)
            ok += 1
        else:
            failed += 1

    print(f"{ok} extracted, {len(images)-ok} failed")

hands.close()

print(f"\nAppending {len(new_rows)} new rows to {CSV_PATH}...")
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(new_rows)

print(f"Done. {len(new_rows)} samples added ({failed} images had no hand detected).")
print("\nNow run: python train_the_model_asl.py")
