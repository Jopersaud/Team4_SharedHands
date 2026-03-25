import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import logging
from PIL import Image
from datetime import datetime
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import cv2
import mediapipe as mp
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cred = credentials.Certificate("../SharedHandsAdminKey.json")

firebase_admin.initialize_app(cred, {
    'storageBucket': 'sharedhands-f232b.appspot.com'
})

db = firestore.client()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================================
# ASL MODEL AND MEDIAPIPE LOADING
# ============================================================================
try:
    asl_model = keras.models.load_model("artifacts/asl_model.keras")
    df = pd.read_csv("artifacts/asl_landmarks.csv", header=None)
    encoder = LabelEncoder()
    encoder.fit(df.iloc[:, 63].values)
    logging.info("Successfully loaded ASL model and label encoder.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)
    logging.info("Successfully initialized MediaPipe Hands.")

except Exception as e:
    logging.error(f"Failed to load models or initialize MediaPipe: {e}")
    asl_model = None
    hands = None

# ============================================================================
# WEBSOCKET TRANSLATION ENDPOINT
# ============================================================================
@socketio.on('video_frame')
def handle_video_frame(data):
    if not asl_model or not hands:
        emit('translation_error', {'error': 'Backend models not loaded'})
        return

    try:
        # Decode the base64 image
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)

        predicted_letter = ''
        confidence_score = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
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
                prediction = asl_model.predict(flat, verbose=0)
                confidence_score = np.max(prediction)
                predicted_letter = encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Encode the processed frame (with or without landmarks) back to base64
        _, buffer = cv2.imencode('.jpeg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        frame_url = f"data:image/jpeg;base64,{encoded_frame}"

        # Send result back to client
        emit('translation_result', {
            'letter': predicted_letter,
            'confidence': float(confidence_score),
            'frame': frame_url
        })

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        emit('translation_error', {'error': 'Error processing frame on server'})


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
            db.collection('users').document(user_record.uid).set(user_profile)
            logging.info(f"Created Firestore profile for user: {user_record.uid}")
            return jsonify({"success": True, "message": "User registered successfully", "uid": user_record.uid, "email": email}), 201
        except Exception as e:
            logging.error(f"Failed to create Firestore profile: {e}")
            try:
                auth.delete_user(user_record.uid)
                logging.info(f"Rolled back: Deleted Auth user {user_record.uid}")
            except: pass
            return jsonify({"success": False, "error": "Failed to create user profile"}), 500
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({"success": False, "error": "Registration failed"}), 500

# ... (keep other routes like /login, /get-users, etc., they are unchanged)

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.json
        uid = data.get('uid')
        if not uid:
            return jsonify({"success": False, "error": "User ID required"}), 400
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            return jsonify({"success": False, "error": "User profile not found"}), 404
        user_data = user_doc.to_dict()
        if user_data.get('accountStatus') != 'active':
            return jsonify({"success": False, "error": "Account is not active"}), 403
        db.collection('users').document(uid).update({'lastLoginAt': firestore.SERVER_TIMESTAMP})
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
    socketio.run(app, debug=True, port=5000)
