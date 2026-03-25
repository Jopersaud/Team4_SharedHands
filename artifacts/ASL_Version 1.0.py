import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import tf_keras as keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd

SMOOTHING_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.6

# --- New MediaPipe Tasks API setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)

hands = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

# --- Load ASL model and encoder ---
model = keras.models.load_model("asl_model.keras", compile=False)

df = pd.read_csv("asl_landmarks.csv", header=None)
encoder = LabelEncoder()
encoder.fit(df.iloc[:, 63].values)

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
display_letter = ""
confidence_score = 0
word = ""
frame_timestamp = 0

print("Welcome... Press Q to quit")
print("Now please show your hand and start signing!!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp += 33  # ~30fps
    results = hands.detect_for_video(mp_image, frame_timestamp)

    detected_letter = None
    hand_detected = len(results.hand_landmarks) > 0

    if hand_detected:
        for hand_landmarks in results.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks, w, h)

        # Use the last detected hand for recognition
        hand_landmarks = results.hand_landmarks[-1]

        # Convert landmarks to list of (x, y, z)
        landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

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

    if display_letter:
        cv2.putText(frame, f"Letter: {display_letter}", (w // 2 - 30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 100), 6)
    else:
        cv2.putText(frame, "Letter: —", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    if display_letter:
        cv2.putText(frame, f"letter: {display_letter} ({confidence_score:.0f}%)", (w // 2 - 80, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "letter: -", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 250), 2)

    cv2.putText(frame, f"word: {word}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 200), 2)

    status = "Hand Detected" if hand_detected else "No Hand Detected"
    color = (0, 230, 100) if hand_detected else (100, 100, 200)
    cv2.putText(frame, status, (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "SPACE = add B = backspace C= clear", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("ASL Testing", frame)

cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL_testing closed.")