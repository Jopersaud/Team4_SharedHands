import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify  


cred = credentials.Certificate("/Users/josh/SPR2026/CIS454/SharedHands/SharedHandsAdminKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

@app.route('/add-user', methods=['POST'])
def add_user():
    try:
        user_data = request.json
        db.collection('users').add(user_data)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)