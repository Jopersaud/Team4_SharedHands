import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter

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


# --- All your recognition functions (unchanged) ---

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

    if not index and not middle and not ring and not pinky and not thumb:
        if t_i < 0.8:
            return "A"

    if index and middle and not ring and not pinky and not thumb:
        return "B"

    if not index and not middle and not ring and not pinky:
        if t_i > 0.5:
            return "C"

    if index and not middle and not ring and not pinky:
        if t_m < 0.6:
            return "D"

    if not index and not middle and not ring and not pinky and not thumb:
        if idx_angle < 90 and mid_angle < 90:
            return "E"

    if index and middle and not ring and not pinky:
        if t_i < 0.5:
            return "F"

    if index and not middle and not ring and not pinky and not thumb:
        return "G"

    if index and middle and not ring and not pinky and not thumb:
        if i_m < 0.3:
            return "H"

    if not index and not middle and not ring and pinky and not thumb:
        return "I"

    if index and middle and not ring and not pinky:
        if t_i < 0.5 and i_m > 0.25:
            return "K"

    if index and not middle and not ring and not pinky and thumb:
        if t_i > 0.6:
            return "L"

    if not index and not middle and not ring and not pinky and thumb:
        return "M"

    if not index and not middle and not ring and not pinky and thumb:
        return "N"

    if not index and not middle and not ring and not pinky and thumb:
        if t_i < 0.4:
            return "O"

    if index and not middle and not ring and not pinky and thumb:
        if t_m < 0.5:
            return "P"

    if index and middle and not ring and not pinky and not thumb:
        if i_m < 0.2:
            return "R"

    if not index and not middle and not ring and not pinky and thumb:
        if idx_angle < 100:
            return "S"

    if not index and not middle and not ring and not pinky and thumb:
        return "T"

    if index and not middle and not ring and pinky and not thumb:
        return "U"

    if index and middle and not ring and not pinky and not thumb:
        if i_m > 0.3:
            return "V"

    if index and middle and ring and not pinky and not thumb:
        return "W"

    if not middle and not ring and not pinky and not thumb:
        if idx_angle < 100 and idx_angle > 40:
            return "X"

    if not index and not middle and not ring and pinky and thumb:
        return "Y"

    return None


# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
current_letter = ""
display_letter = ""
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
        hand_landmarks = results.hand_landmarks[0]

        # Draw landmarks
        draw_hand_landmarks(frame, hand_landmarks, w, h)

        # Convert landmarks to list of (x, y, z)
        landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

        # Run recognition
        detected_letter = recognize_sign(landmarks_list)

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

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ASL_testing Recognizer", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if display_letter:
        cv2.putText(frame, f"Letter: {display_letter}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 4)
    else:
        cv2.putText(frame, "Letter: —", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    status = "Hand Detected" if hand_detected else "No Hand Detected"
    color = (0, 230, 100) if hand_detected else (100, 100, 200)
    cv2.putText(frame, status, (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("ASL Testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL_testing closed.")