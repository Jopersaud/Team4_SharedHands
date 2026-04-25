import cv2
import mediapipe as mp
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import deque, Counter
import json
import time

SMOOTHING_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.6

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

model = keras.models.load_model("asl_model.keras")


class TransformerBLock(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = keras.layers.Dense(ff_dim, activation="relu")
        self.ff2 = None  # fuck me

    def build(self, input_shape):
        self.ff2 = keras.layers.Dense(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=False):
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        ff_output = self.ff1(x)
        ff_output = self.ff2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        x = self.norm2(x + ff_output)

        return x


transformer_model = keras.models.load_model(
    "asl_transformer.keras",
    custom_objects={"TransformerBLock": TransformerBLock}
)
transformer_classes = np.load("Transformer_classes.npy", allow_pickle=True)
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
sequence_buffer = deque(maxlen=30)
transformer_prediction = ""
transformer_confidence = 0
frame_count = 0
prev_time = time.time()

print("Welcome... Press Q to quit")
print("Now please show your hand and start signing!!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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
        flat_seq = [coord for point in landmarks_list for coord in point]
        sequence_buffer.append(flat_seq)
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

    if len(sequence_buffer) == 30 and frame_count % 30 == 0:
        seq_input = np.array(sequence_buffer).reshape(1, 30, 63)
        trans_pred = transformer_model.predict(seq_input, verbose=0)
        transformer_confidence = np.max(trans_pred) * 100
        transformer_prediction = transformer_classes[np.argmax(trans_pred)]

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
    if transformer_prediction:
        cv2.putText(frame, f"Motion: {transformer_prediction} ({transformer_confidence}%)", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

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

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    # Show frame
    cv2.imshow("ASL Testing", frame)

cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL_testing closed.")
