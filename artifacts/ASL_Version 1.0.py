import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import deque, Counter

SMOOTHING_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.6

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

model = keras.models.load_model("asl_model.keras")

df = pd.read_csv("asl_landmarks.csv", header=None)
encoder = LabelEncoder()
encoder.fit(df.iloc[:, 63].values)

# This is where the camera functions relies on the previous functions

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
display_letter = ""
confidence_score = 0
word = ""

print("Welcome... Press Q to quit")
print("Now please show your hand and start signing!!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)

    detected_letter = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Convert landmarks to list of (x, y, z)
        landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # Run recognition
        flat = np.array([coord for point in landmarks_list for coord in point]).reshape(1, -1)
        prediction = model.predict(flat, verbose=0)
        confidence_score = np.max(prediction) * 100
        detected_letter = encoder.inverse_transform([np.argmax(prediction)])[0]

    prediction_buffer.append(detected_letter)

    if len(prediction_buffer) == SMOOTHING_WINDOW:
        votes = Counter(p for p in prediction_buffer if p is not None)
        if votes:
            best_letter, best_count = votes.most_common(1)[0]
            confidence = best_count / SMOOTHING_WINDOW
            if confidence >= CONFIDENCE_THRESHOLD:
                display_letter = best_letter
            else:
                display_letter = ""
        else:
            display_letter = ""

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and display_letter:
        word += display_letter
        print(f"You wrote: {word}")
    if key == ord("b"):
        word = word[:-1]
    if key == ord("c"):
        word = ""
    if key == ord("q"):
        break

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ASL_testing Recognizer", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Detected letter — big and bold
    if display_letter:
        cv2.putText(frame, f"Letter: {display_letter}", (w // 2 - 30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 100), 6)
    else:
        cv2.putText(frame, "Letter: —", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    # confidence
    if display_letter:
        cv2.putText(frame, f"letter: {display_letter} ({confidence_score:.0f}%)", (w // 2 - 80, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "letter: -", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 250), 2)

    cv2.putText(frame, f"word: {word}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 200), 2)

    # Hand detection status
    status = "Hand Detected" if results.multi_hand_landmarks else "No Hand Detected"
    color = (0, 230, 100) if results.multi_hand_landmarks else (100, 100, 200)
    cv2.putText(frame, status, (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "SPACE = add B = backspace C= clear", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Show frame
    cv2.imshow("ASL Testing", frame)

cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL_testing closed.")
