import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



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

        ### Draw landmarks on frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        ### Convert landmarks to list of (x, y, z)
        landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]



    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ASL testing", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


    ### Hand detection status
    status = "Hand Detected" if results.multi_hand_landmarks else "No Hand Detected"
    color = (0, 230, 100) if results.multi_hand_landmarks else (100, 100, 200)
    cv2.putText(frame, status, (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    ### Shows the frame
    cv2.imshow("ASL Testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("ASL testing is closed.")


