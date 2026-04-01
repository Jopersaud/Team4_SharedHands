import eventlet
import eventlet.tpool
eventlet.monkey_patch(os=True, select=True, socket=True, thread=False, time=True)

import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import logging
from datetime import datetime
import numpy as np
import tf_keras as keras
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter
import pandas as pd
import cv2
import mediapipe as mp
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# FIREBASE ADMIN SETUP
# ============================================================================
KEY_PATH = os.environ.get(
    "FIREBASE_KEY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sharedhandsadminkey.json")
)
cred = credentials.Certificate(KEY_PATH)

firebase_admin.initialize_app(cred, {
    'storageBucket': 'sharedhands-17f7b.appspot.com'
})

db = firestore.client()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

# ============================================================================
# ASL MODEL AND MEDIAPIPE TASKS API SETUP
# ============================================================================
SMOOTHING_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.6

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

_BASE = os.path.dirname(os.path.abspath(__file__))
_TASK_PATH  = os.path.join(_BASE, "hand_landmarker.task")
_MODEL_PATH = os.path.join(_BASE, "asl_model.keras")
_CSV_PATH   = os.path.join(_BASE, "asl_landmarks.csv")

try:
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    _options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_TASK_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    hand_landmarker = HandLandmarker.create_from_options(_options)

    asl_model = keras.models.load_model(_MODEL_PATH, compile=False)
    df = pd.read_csv(_CSV_PATH, header=None)
    encoder = LabelEncoder()
    encoder.fit(df.iloc[:, 63].values)
    logging.info("ASL model, encoder, and HandLandmarker loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models or initialize HandLandmarker: {e}")
    hand_landmarker = None
    asl_model = None
    encoder = None

# Per-client prediction buffers { sid: deque(maxlen=15) }
client_buffers = {}


def draw_hand_landmarks(frame, landmarks, w, h):
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)


# ============================================================================
# WEBSOCKET HANDLERS
# ============================================================================
@socketio.on('connect')
def handle_connect():
    client_buffers[request.sid] = deque(maxlen=SMOOTHING_WINDOW)
    logging.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    client_buffers.pop(request.sid, None)
    logging.info(f"Client disconnected, buffer cleaned: {request.sid}")


@socketio.on('video_frame')
def handle_video_frame(data):
    if not hand_landmarker or not asl_model:
        emit('translation_error', {'error': 'Backend models not loaded'})
        return

    sid = request.sid
    if sid not in client_buffers:
        client_buffers[sid] = deque(maxlen=SMOOTHING_WINDOW)

    try:
        def process():
            # Decode base64 frame
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]

            # Run MediaPipe Tasks API (IMAGE mode — stateless, no timestamp needed)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = hand_landmarker.detect(mp_image)

            detected_letter = None
            if results.hand_landmarks:
                for hand_lms in results.hand_landmarks:
                    draw_hand_landmarks(frame, hand_lms, w, h)
                lms = results.hand_landmarks[-1]
                flat = np.array([(lm.x, lm.y, lm.z) for lm in lms]).reshape(1, -1)
                prediction = asl_model.predict(flat, verbose=0)
                detected_letter = encoder.inverse_transform([np.argmax(prediction)])[0]

            client_buffers[sid].append(detected_letter)

            display_letter = ''
            smoothed_confidence = 0.0
            buf = client_buffers[sid]
            if len(buf) == SMOOTHING_WINDOW:
                votes = Counter(p for p in buf if p is not None)
                if votes:
                    best_letter, best_count = votes.most_common(1)[0]
                    smoothed_confidence = best_count / SMOOTHING_WINDOW
                    if smoothed_confidence >= CONFIDENCE_THRESHOLD:
                        display_letter = best_letter

            _, buffer = cv2.imencode('.jpeg', frame)
            frame_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            return display_letter, float(smoothed_confidence), frame_url

        display_letter, smoothed_confidence, frame_url = eventlet.tpool.execute(process)

        emit('translation_result', {
            'letter': display_letter,
            'confidence': smoothed_confidence,
            'frame': frame_url
        })

    except Exception as e:
        logging.error(f"Error processing frame for {sid}: {e}")
        emit('translation_error', {'error': 'Error processing frame on server'})


# ============================================================================
# AUTH ROUTES
# ============================================================================
@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        try:
            user_record = auth.create_user(
                email=email,
                password=password,
                email_verified=False
            )
            logging.info(f"Created Firebase Auth user: {user_record.uid}")
        except auth.EmailAlreadyExistsError:
            return jsonify({"success": False, "error": "Email already registered"}), 400
        except Exception as e:
            logging.error(f"Failed to create Firebase Auth user: {e}")
            return jsonify({"success": False, "error": "Failed to create user account"}), 500

        try:
            user_profile = {
                'uid': user_record.uid, 'email': email,
                'createdAt': firestore.SERVER_TIMESTAMP, 'lastLoginAt': firestore.SERVER_TIMESTAMP,
                'accountStatus': 'active', 'subscriptionTier': 'free', 'subscriptionStatus': 'active',
                'subscriptionId': None, 'premiumFeaturesEnabled': False,
                'preferences': {
                    'outputLanguage': 'en', 'signLanguageType': 'ASL',
                    'camera': {'defaultCamera': 'front', 'resolution': 'medium', 'fps': 30},
                },
                'organizationId': None, 'organizationRole': None,
            }
            eventlet.tpool.execute(lambda: db.collection('users').document(user_record.uid).set(user_profile))
            logging.info(f"Created Firestore profile for user: {user_record.uid}")
            return jsonify({"success": True, "message": "User registered successfully", "uid": user_record.uid, "email": email}), 201
        except Exception as e:
            logging.error(f"Failed to create Firestore profile: {e}")
            try:
                auth.delete_user(user_record.uid)
                logging.info(f"Rolled back: Deleted Auth user {user_record.uid}")
            except Exception:
                pass
            return jsonify({"success": False, "error": "Failed to create user profile"}), 500
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({"success": False, "error": "Registration failed"}), 500


@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.json
        uid = data.get('uid')
        if not uid:
            return jsonify({"success": False, "error": "User ID required"}), 400
        user_doc = eventlet.tpool.execute(lambda: db.collection('users').document(uid).get())
        if not user_doc.exists:
            return jsonify({"success": False, "error": "User profile not found"}), 404
        user_data = user_doc.to_dict()
        if user_data.get('accountStatus') != 'active':
            return jsonify({"success": False, "error": "Account is not active"}), 403
        eventlet.tpool.execute(lambda: db.collection('users').document(uid).update({'lastLoginAt': firestore.SERVER_TIMESTAMP}))
        logging.info(f"User logged in: {uid}")
        return jsonify({"success": True, "user": user_data}), 200
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({"success": False, "error": "Login failed"}), 500


@app.route('/get-users', methods=['GET'])
def get_users():
    try:
        users_ref = db.collection('users')
        users = [doc.to_dict() for doc in users_ref.stream()]
        return jsonify({"success": True, "users": users, "count": len(users)}), 200
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
