import firebase_admin
from firebase_admin import credentials, firestore, storage
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from PIL import Image
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Using hardcoded path for Firebase credentials as requested.
# Please be aware that for production environments, using environment variables
# or other secure methods to manage credentials is highly recommended.
cred = credentials.Certificate("/Users/josh/SPR2026/CIS454/SharedHands/SharedHandsAdminKey.json")
# NOTE: You may need to replace 'your-project-id.appspot.com' with your actual Firebase Storage bucket name.
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sharedhands-f232b.appspot.com'
})

db = firestore.client()

app = Flask(__name__)

# This allows all origins for all routes, which is convenient for development.
# For production, you should restrict this to your frontend's domain.
CORS(app)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        logging.error("No image file provided in request to /upload-image")
        return jsonify({"success": False, "error": "No image file in request"}), 400

    image_file = request.files['image']

    try:
        # Validate that it's a valid image file
        Image.open(image_file).verify()
        # Reset stream position after verification
        image_file.seek(0)
    except Exception as e:
        logging.error(f"Invalid image file provided: {e}")
        return jsonify({"success": False, "error": f"Invalid image file: {e}"}), 400

    try:
        # Create a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"captures/single/{timestamp}.jpg"
        
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        
        # Upload the file
        blob.upload_from_file(image_file, content_type='image/jpeg')
        
        # Make the file publicly accessible
        blob.make_public()
        
        logging.info(f"Successfully uploaded single image: {filename}")
        return jsonify({"success": True, "url": blob.public_url}), 200
    except Exception as e:
        logging.error(f"Failed to upload single image: {e}")
        return jsonify({"success": False, "error": "Failed to upload image to storage"}), 500

@app.route('/upload-sequence', methods=['POST'])
def upload_sequence():
    if 'images' not in request.files:
        logging.error("No image files provided in request to /upload-sequence")
        return jsonify({"success": False, "error": "No 'images' file part in request"}), 400

    image_files = request.files.getlist('images')
    
    if not image_files or image_files[0].filename == '':
        logging.error("No images selected for upload in /upload-sequence")
        return jsonify({"success": False, "error": "No images selected for upload"}), 400

    sequence_id = f"sequence_{firestore.SERVER_TIMESTAMP}"
    bucket = storage.bucket()
    
    success_urls = []
    failed_files = []

    for i, image_file in enumerate(image_files):
        try:
            # Validate image
            Image.open(image_file).verify()
            image_file.seek(0)
        except Exception as e:
            logging.warning(f"Invalid image file in sequence {sequence_id}: {image_file.filename} - {e}")
            failed_files.append({"filename": image_file.filename, "error": f"Invalid image file: {e}"})
            continue

        try:
            filename = f"captures/sequences/{sequence_id}/frame_{i+1}.jpg"
            blob = bucket.blob(filename)
            blob.upload_from_file(image_file, content_type='image/jpeg')
            blob.make_public()
            success_urls.append(blob.public_url)
        except Exception as e:
            logging.error(f"Failed to upload file {image_file.filename} in sequence {sequence_id}: {e}")
            failed_files.append({"filename": image_file.filename, "error": "Failed to upload to storage"})

    logging.info(f"Upload for sequence {sequence_id} complete. Success: {len(success_urls)}, Failed: {len(failed_files)}")
    return jsonify({
        "success": True,
        "sequence_id": sequence_id,
        "uploaded_urls": success_urls,
        "failed_files": failed_files
    }), 200

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
