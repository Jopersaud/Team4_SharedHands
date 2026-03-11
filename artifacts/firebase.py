import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cred = credentials.Certificate("/Users/josh/SPR2026/CIS454/SharedHands/SharedHandsAdminKey.json")  

firebase_admin.initialize_app(cred, {
    'storageBucket': 'sharedhands-f232b.appspot.com'
})

db = firestore.client()

app = Flask(__name__)
CORS(app)


@app.route('/register', methods=['POST'])
def register_user():

    try:
        data = request.json
        
        # Extract required fields
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
            return jsonify({
                "success": False,
                "error": "Email already registered"
            }), 400
        except Exception as e:
            logging.error(f"Failed to create Firebase Auth user: {e}")
            return jsonify({
                "success": False,
                "error": "Failed to create user account"
            }), 500
        
        try:
            user_profile = {
                # Identity & Authentication
                'uid': user_record.uid,
                'email': email,
                
                # Account Management
                'createdAt': firestore.SERVER_TIMESTAMP,
                'lastLoginAt': firestore.SERVER_TIMESTAMP,
                'accountStatus': 'active',
                
                # Subscription & Tier
                'subscriptionTier': 'free',
                'subscriptionStatus': 'active',
                'subscriptionId': None,
                'premiumFeaturesEnabled': False,
                
                # User Preferences (with defaults)
                'preferences': {
                    'outputLanguage': 'en',
                    'signLanguageType': 'ASL',
                    'camera': {
                        'defaultCamera': 'front',
                        'resolution': 'medium',
                        'fps': 30
                    },
                },
                # Organization (for enterprise users)
                'organizationId': None,
                'organizationRole': None,
            }
            
            db.collection('users').document(user_record.uid).set(user_profile)
            
            logging.info(f"Created Firestore profile for user: {user_record.uid}")
            
            return jsonify({
                "success": True,
                "message": "User registered successfully",
                "uid": user_record.uid,
                "email": email
            }), 201
            
        except Exception as e:
            logging.error(f"Failed to create Firestore profile: {e}")
            try:
                auth.delete_user(user_record.uid)
                logging.info(f"Rolled back: Deleted Auth user {user_record.uid}")
            except:
                pass
            
            return jsonify({
                "success": False,
                "error": "Failed to create user profile"
            }), 500
            
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({
            "success": False,
            "error": "Registration failed"
        }), 500

@app.route('/login', methods=['POST'])
def login_user():
    """
    Verify user credentials and return user data
    Note: Password verification happens via Firebase Auth on frontend
    This endpoint is for additional backend logic after successful auth
    """
    try:
        data = request.json
        uid = data.get('uid')  # Frontend sends UID after Firebase Auth login
        
        if not uid:
            return jsonify({
                "success": False,
                "error": "User ID required"
            }), 400
        
        # Get user profile from Firestore
        user_doc = db.collection('users').document(uid).get()
        
        if not user_doc.exists:
            return jsonify({
                "success": False,
                "error": "User profile not found"
            }), 404
        
        user_data = user_doc.to_dict()
        
        # Check account status
        if user_data.get('accountStatus') != 'active':
            return jsonify({
                "success": False,
                "error": "Account is not active"
            }), 403
        
        # Update last login time
        db.collection('users').document(uid).update({
            'lastLoginAt': firestore.SERVER_TIMESTAMP,
            'security.failedLoginAttempts': 0  # Reset failed attempts
        })
        
        logging.info(f"User logged in: {uid}")
        
        return jsonify({
            "success": True,
            "user": user_data
        }), 200
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({
            "success": False,
            "error": "Login failed"
        }), 500



@app.route('/get-users', methods=['GET'])
def get_users():
    """Get all users (for admin dashboard)"""
    try:
        users_ref = db.collection('users')
        users = []
        
        for doc in users_ref.stream():
            user_data = doc.to_dict()
            
            # Remove sensitive data
            if 'security' in user_data:
                user_data['security'] = {
                    'twoFactorEnabled': user_data['security'].get('twoFactorEnabled', False)
                }
            
            users.append(user_data)
        
        return jsonify({
            "success": True,
            "users": users,
            "count": len(users)
        }), 200
        
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ============================================================================
# UPDATE USER PROFILE
# ============================================================================

@app.route('/user/<uid>', methods=['PATCH'])
def update_user(uid):
    """Update user profile"""
    try:
        updates = request.json
        
        # Don't allow updating certain fields
        protected_fields = ['uid', 'email', 'createdAt', 'subscriptionTier']
        for field in protected_fields:
            if field in updates:
                del updates[field]
        
        # Add update timestamp
        updates['updatedAt'] = firestore.SERVER_TIMESTAMP
        
        db.collection('users').document(uid).update(updates)
        
        logging.info(f"Updated user profile: {uid}")
        
        return jsonify({
            "success": True,
            "message": "User updated successfully"
        }), 200
        
    except Exception as e:
        logging.error(f"Error updating user: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to update user"
        }), 500

# ============================================================================
# IMAGE UPLOAD ENDPOINTS (Your existing code - unchanged)
# ============================================================================

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        logging.error("No image file provided in request to /upload-image")
        return jsonify({"success": False, "error": "No image file in request"}), 400

    image_file = request.files['image']

    try:
        Image.open(image_file).verify()
        image_file.seek(0)
    except Exception as e:
        logging.error(f"Invalid image file provided: {e}")
        return jsonify({"success": False, "error": f"Invalid image file: {e}"}), 400

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"captures/single/{timestamp}.jpg"
        
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_file(image_file, content_type='image/jpeg')
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

    sequence_id = f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bucket = storage.bucket()
    
    success_urls = []
    failed_files = []

    for i, image_file in enumerate(image_files):
        try:
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
@app.route('/update-subscription', methods=['POST'])
def update_subscription():
    try:
        data = request.json
        
        # Validate required fields
        uid = data.get('uid')
        subscription_tier = data.get('subscriptionTier')
        
        if not uid:
            return jsonify({
                "success": False,
                "error": "User ID (uid) is required"
            }), 400
        
        if not subscription_tier:
            return jsonify({
                "success": False,
                "error": "subscriptionTier is required"
            }), 400
        
        # Validate subscription tier
        valid_tiers = ['free', 'premium', 'enterprise']
        if subscription_tier not in valid_tiers:
            return jsonify({
                "success": False,
                "error": f"Invalid subscription tier. Must be one of: {', '.join(valid_tiers)}"
            }), 400
        
        # Optional: subscription status (defaults to 'active')
        subscription_status = data.get('subscriptionStatus', 'active')
        valid_statuses = ['active', 'cancelled', 'expired', 'trialing']
        if subscription_status not in valid_statuses:
            return jsonify({
                "success": False,
                "error": f"Invalid subscription status. Must be one of: {', '.join(valid_statuses)}"
            }), 400
        
        # Check if user exists
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({
                "success": False,
                "error": f"User not found: {uid}"
            }), 404
        
        # Determine premium features based on tier
        premium_features_enabled = subscription_tier in ['premium', 'enterprise']
        
        # Prepare updates
        updates = {
            'subscriptionTier': subscription_tier,
            'subscriptionStatus': subscription_status,
            'premiumFeaturesEnabled': premium_features_enabled,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        # Update Firestore
        user_ref.update(updates)
        
        logging.info(f"Updated subscription for user {uid}: {subscription_tier} ({subscription_status})")
        
        # Get updated user data
        updated_user = user_ref.get().to_dict()
        
        return jsonify({
            "success": True,
            "message": f"Subscription updated to {subscription_tier}",
            "user": {
                "uid": uid,
                "email": updated_user.get('email'),
                "subscriptionTier": updated_user.get('subscriptionTier'),
                "subscriptionStatus": updated_user.get('subscriptionStatus'),
                "premiumFeaturesEnabled": updated_user.get('premiumFeaturesEnabled')
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Error updating subscription: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to update subscription: {str(e)}"
        }), 500
# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)