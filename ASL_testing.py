import cv2
import mediapipe as mp
import numpy as np
from collections import deque

SMOOTHING_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.6

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 4,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

def get_finger_states(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]

    states = []

    wrist_x = landmarks[0][0]
    thumb_tip_x = landmarks[4][0]
    thumb_pip_x = landmarks[2][0]

    wrist_x = landmarks[4][2]
    thumb_tip_x = landmarks[4][0]

    mid_tip_x = landmarks[12][0]
    if wrist_x < mid_tip_x:
        states.append(thumb_tip_x < thumb_pip_x)
    else:
        states.append(thumb_tip_x > thumb_pip_x)

    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        states.append(landmarks[tip][1] < landmarks[pip][1])

    return states

def get_finger_angles(landmarks):
    def angle_between(p1, p2, p3):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    idx_angle = angle_between(landmarks[8], landmarks[6], landmarks[5])

    mid_angle = angle_between(landmarks[12], landmarks[10], landmarks[9])

    ring_angle = angle_between(landmarks[16], landmarks[14], landmarks[13])

    pink_angle = angle_between(landmarks[20], landmarks[18], landmarks[17])

    return idx_angle, mid_angle, ring_angle, pink_angle

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def recognize_sign(landmarks_list):

    lm = landmarks_list
    fingers = get_finger_states(lm)
    thumb, index, middle, ring, pinky = fingers
    idx_angle, mid_angle, ring_angle, pink_angle = get_finger_angles(lm)

    thumb_to_index = distance(lm[4], lm[8])
    index_to_middle= distance(lm[8], lm[12])
    thumb_to_middle = distance(lm[4], lm[12])
    index_to_pinky = distance(lm[8], lm[20])
    thumb_to_pinky = distance(lm[4], lm[20])
    palm_size = distance(lm[0], lm[9])

    t_i = thumb_to_index / (palm_size + 1e-6)
    i_m = index_to_middle / (palm_size + 1e-6)
    t_m = thumb_to_middle / (palm_size + 1e-6)
    i_p = index_to_pinky / (palm_size + 1e-6)
    t_p = thumb_to_pinky / (palm_size + 1e-6)


# A
    if not index and not middle and not ring and not pinky and not thumb:
        if t_i < 0.8:
            return "A"

    # B
    if index and middle and not ring and not pinky and not thumb:
        return "B"

    # C
    if not index and not middle and not ring and not pinky:
        if t_i > 0.5:
            return "C"

    # D
    if index and not middle and not ring and not pinky:
        if t_m < 0.6:
            return "D"

    # E
    if not index and not middle and not ring and not pinky and not thumb:
        if idx_angle < 90 and mid_angle < 90:
            return "E"

    # F
    if index and middle and not ring and not pinky:
        if t_i < 0.5:
            return "F"

    # G
    if index and not middle and not ring and not pinky and not thumb:
        return "G"

    # H
    if index and middle and not ring and not pinky and not thumb:
        if i_m < 0.3:
            return "H"

    # I
    if not index and not middle and not ring and pinky and not thumb:
        return "I"

    # K
    if index and middle and not ring and not pinky:
        if t_i < 0.5 and i_m > 0.25:
            return "K"

    # L
    if index and not middle and not ring and not pinky and thumb:
        if t_i > 0.6:
            return "L"

    # M
    if not index and not middle and not ring and not pinky and thumb:
        return "M"

    # N
    if not index and not middle and not ring and not pinky and thumb:
        return "N"

    # O
    if not index and not middle and not ring and not pinky and thumb:
        if t_i < 0.4:
            return "O"

    # P
    if index and not middle and not ring and not pinky and thumb:
        if t_m < 0.5:
            return "P"

    # R
    if index and middle and not ring and not pinky and not thumb:
        if i_m < 0.2:
            return "R"

    # S
    if not index and not middle and not ring and not pinky and thumb:
        if idx_angle < 100:
            return "S"

    # T
    if not index and not middle and not ring and not pinky and thumb:
        return "T"

    # U
    if index and not middle and not ring and pinky and not thumb:
        return "U"

    # V
    if index and middle and not ring and not pinky and not thumb:
        if i_m > 0.3:
            return "V"

    # W
    if index and middle and ring and not pinky and not thumb:
        return "W"

    # X
    if not middle and not ring and not pinky and not thumb:
        if idx_angle < 100 and idx_angle > 40:
            return "X"

    # Y
    if not index and not middle and not ring and pinky and thumb:
        return "Y"

    return None


# This is where the camera functions relies on the previous functions

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
current_letter = ""
display_letter = ""

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
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on frame
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
        detected_letter = recognize_sign(landmarks_list)

    prediction_buffer.append(detected_letter)

    if len(prediction_buffer) == SMOOTHING_WINDOW:
        from collections import Counter

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

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ASL_testing Recognizer", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Detected letter — big and bold
    if display_letter:
        cv2.putText(frame, f"Letter: {display_letter}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 4)
    else:
        cv2.putText(frame, "Letter: —", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    # Hand detection status
    status = "Hand Detected" if results.multi_hand_landmarks else "No Hand Detected"
    color = (0, 230, 100) if results.multi_hand_landmarks else (100, 100, 200)
    cv2.putText(frame, status, (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show frame
    cv2.imshow("ASL Testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL_testing closed.")
