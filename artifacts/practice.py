import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import zipfile

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5

)
output_rows = []
failed = 0

with zipfile.ZipFile("SignAlphaSet.zip", "r") as z:
    z.extractall("SignAlphaSet")


for letter in os.listdir("SignAlphaSet/SignAlphaSet"):
    folder_path = os.path.join("SignAlphaSet/SignAlphaSet", letter)
    if not os.path.isdir(folder_path):
        continue

    label = letter.upper()
    print(f"Processing: {label}")

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            row = []
            for point in lm:
                row.extend([point.x, point.y, point.z])
            row.append(label)
            output_rows.append(row)

        else:
            failed += 1


with open("asl_landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)


print(f"FINITO {len(output_rows)} samples has been saved")
print(f"{failed} images have occured")
hands.close()