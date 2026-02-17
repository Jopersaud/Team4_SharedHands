import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Using hardcoded path for Firebase credentials as requested.
# Please be aware that for production environments, using environment variables
# or other secure methods to manage credentials is highly recommended.
cred = credentials.Certificate("/Users/josh/SPR2026/CIS454/SharedHands/SharedHandsAdminKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

CORS(app, resources={r"/register": {"origins": "http://localhost:3000"}})

@app.route('/register', methods=['POST'])
def add_user():
    try:
        user_data = request.json
        db.collection('users').add(user_data)
        return jsonify({"success": True, "message": "User created successfully"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/get-users', methods=['GET'])
def get_users():
    try:
        users_ref = db.collection('users')
        users = [doc.to_dict() for doc in users_ref.stream()]
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
