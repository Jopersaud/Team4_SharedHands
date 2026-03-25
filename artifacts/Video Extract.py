import cv2
import mediapipe as mp
import numpy as np
import os
import json
import zipfile


with zipfile.ZipFile("ASL_dynamic.zip", "r") as z:
    z.extractall("ASL_dynamic")
SEQUENCE_LENGTH = 30  # how many frames per sequence we'll use

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

sequences = []  # will hold {label, frames: [[63 numbers], ...]}
failed = 0
saved = 0

dataset_path = "ASL_dynamic/ASL_dynamic"

for sign_label in os.listdir(dataset_path):
    sign_path = os.path.join(dataset_path, sign_label)
    if not os.path.isdir(sign_path):
        continue

    print(f"Processing sign: {sign_label}")

    # Find all clip_frames folders inside this sign folder
    clip_folders = [
        f for f in os.listdir(sign_path)
        if os.path.isdir(os.path.join(sign_path, f)) and "frames" in f
    ]

    for clip_folder in clip_folders:
        clip_path = os.path.join(sign_path, clip_folder)

        # Get all jpg frames sorted by name
        frame_files = sorted([
            f for f in os.listdir(clip_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

        if len(frame_files) < 10:  # skip clips that are too short
            continue

        # Sample exactly SEQUENCE_LENGTH frames evenly from the clip
        indices = np.linspace(0, len(frame_files) - 1, SEQUENCE_LENGTH, dtype=int)
        selected_frames = [frame_files[i] for i in indices]

        landmark_sequence = []
        valid = True

        for frame_file in selected_frames:
            img_path = os.path.join(clip_path, frame_file)
            img = cv2.imread(img_path)
            if img is None:
                valid = False
                break

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                frame_landmarks = []
                for point in lm:
                    frame_landmarks.extend([point.x, point.y, point.z])
                landmark_sequence.append(frame_landmarks)
            else:
                # No hand detected in this frame — skip whole clip
                valid = False
                break

        if valid and len(landmark_sequence) == SEQUENCE_LENGTH:
            sequences.append({
                "label": sign_label,
                "sequence": landmark_sequence
            })
            saved += 1
        else:
            failed += 1

hands.close()

# Save as JSON (sequences are 3D so JSON is easier than CSV here)
with open("asl_sequences.json", "w") as f:
    json.dump(sequences, f)


with open("asl_sequences.json", "r") as f:
    data = json.load(f)

labels = sorted(set(item["label"] for item in data))
print(f"Total sequences: {len(data)}")
print(f"Signs found: {labels}")
print(f"Count per sign:")
from collections import Counter
counts = Counter(item["label"] for item in data)
for label, count in sorted(counts.items()):
    print(f"  {label}: {count} sequences")

print(f"Done! {saved} sequences saved, {failed} skipped")
