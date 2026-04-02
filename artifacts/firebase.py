import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# FIREBASE ADMIN SETUP
# ============================================================================
KEY_PATH = os.environ.get(
    "FIREBASE_KEY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SharedHandsAdminKey.json")
)
cred = credentials.Certificate(KEY_PATH)

firebase_admin.initialize_app(cred, {
    'storageBucket': 'sharedhands-f232b.appspot.com'
})

db = firestore.client()

app = Flask(__name__)
CORS(app)


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
            db.collection('users').document(user_record.uid).set(user_profile)
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
    app.run(debug=True, host='0.0.0.0', port=5000)
