# Backend Flask App's Role in Video Feed Processing


## 1. SharedHands Backend Architecture

### Overview

The Flask backend serves as the **"AI processing engine"** for SharedHands, handling all computationally intensive tasks that cannot or should not be performed in the browser.

### Core Responsibilities

```
BACKEND FLASK APP (SharedHands)
├── Frame Reception & Validation
├── User Authentication & Authorization
├── Rate Limiting by Subscription Tier
├── Image Preprocessing
├── Hand Landmark Extraction (MediaPipe)
├── Neural Network Inference (TensorFlow)
├── Translation History Storage (Firestore)
└── Response Formatting & Error Handling
```

---

## 2. Complete Video Processing Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Browser)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Camera Access (getUserMedia API)                                   │
│     - Request permissions                                              │
│     - Start video stream at 30 FPS                                     │
│                                                                         │
│  2. Video Display                                                      │
│     - Show live feed to user                                           │
│     - Mirror mode for front camera                                     │
│                                                                         │
│  3. Frame Capture (2-5 FPS)                                            │
│     - Extract frame from video element                                 │
│     - Draw to canvas                                                   │
│     - Convert to base64 JPEG/PNG                                       │
│     - ~20-30ms per frame                                               │
│                                                                         │
│  4. HTTP POST Request                                                  │
│     POST /api/v1/video/process                                         │
│     {                                                                  │
│       "userId": "user123",                                             │
│       "sessionId": "session_abc",                                      │
│       "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",            │
│       "timestamp": "2026-02-17T10:30:00Z"                             │
│     }                                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        BACKEND FLASK APP (Server)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  5. Request Reception                                                  │
│     @app.route('/api/v1/video/process', methods=['POST'])             │
│     - Receive JSON payload                                             │
│     - Extract frame data                                               │
│     - ~10ms                                                            │
│                                                                         │
│  6. Authentication & Validation                                        │
│     - Verify JWT token                                                 │
│     - Check user subscription tier                                     │
│     - Validate frame size (< 5MB)                                      │
│     - Check rate limits (10/60/300 req/min)                           │
│     - ~20ms                                                            │
│                                                                         │
│  7. Frame Decoding                                                     │
│     - Decode base64 → bytes                                            │
│     - Convert to PIL Image                                             │
│     - Convert to NumPy array                                           │
│     - ~30-50ms                                                         │
│                                                                         │
│  8. Image Preprocessing                                                │
│     - Resize to 640x480 (model input size)                            │
│     - Convert BGR → RGB (if needed)                                    │
│     - Normalize pixel values (0-255 → 0-1)                            │
│     - Quality validation (blur, lighting)                              │
│     - ~20-30ms                                                         │
│                                                                         │
│  9. MediaPipe Hand Detection                                           │
│     import mediapipe as mp                                             │
│     mp_hands = mp.solutions.hands                                      │
│                                                                         │
│     with mp_hands.Hands() as hands:                                    │
│         results = hands.process(frame)                                 │
│         landmarks = extract_landmarks(results)                         │
│                                                                         │
│     - Detect hands in frame                                            │
│     - Extract 21 landmarks × 2 hands × 3 coords = 126 features       │
│     - Calculate confidence scores                                      │
│     - ~150-200ms                                                       │
│                                                                         │
│ 10. Landmark Preprocessing                                             │
│     - Normalize landmarks (scale/translation invariant)                │
│     - Calculate geometric features (angles, distances)                 │
│     - Create sequence buffer (30 frames)                               │
│     - Pad or truncate to model input size                             │
│     - ~10-20ms                                                         │
│                                                                         │
│ 11. Neural Network Inference                                           │
│     model = load_model('asl_model_v1.0.0.h5')  # Cached              │
│     prediction = model.predict(landmarks_normalized)                   │
│                                                                         │
│     - LSTM/Transformer forward pass                                    │
│     - Softmax activation                                               │
│     - Get top-k predictions                                            │
│     - ~150-200ms (CPU) or ~50-100ms (GPU)                             │
│                                                                         │
│ 12. Post-Processing                                                    │
│     - Map prediction index to gesture name                             │
│     - Apply confidence thresholding                                    │
│     - Sequence smoothing (reduce flicker)                              │
│     - Format for translation                                           │
│     - ~10-20ms                                                         │
│                                                                         │
│ 13. Database Storage (Async)                                           │
│     firestore_service.create_translation(                             │
│         user_id=user_id,                                               │
│         translated_text=gesture_name,                                  │
│         confidence_score=confidence,                                   │
│         timestamp=datetime.utcnow()                                    │
│     )                                                                  │
│     - Save to Firestore (non-blocking)                                │
│     - Update user statistics                                           │
│                                                                         │
│ 14. Response Formatting                                                │
│     {                                                                  │
│       "success": true,                                                 │
│       "translation": {                                                 │
│         "translatedText": "Hello, how are you?",                      │
│         "detectedGestures": ["hello", "how", "are", "you"],          │
│         "confidenceScore": 0.94,                                      │
│         "perGestureConfidence": [0.98, 0.92, 0.95, 0.91]            │
│       },                                                               │
│       "handData": {                                                    │
│         "handsDetected": 2,                                            │
│         "visibility": 0.95                                             │
│       },                                                               │
│       "performance": {                                                 │
│         "totalProcessingTime": 387,  // milliseconds                  │
│         "modelVersion": "v1.2.3"                                      │
│       }                                                                │
│     }                                                                  │
│     - ~10-20ms                                                         │
│                                                                         │
│ 15. HTTP Response                                                      │
│     return jsonify(result), 200                                        │
│     - Send JSON back to frontend                                       │
│     - ~10-20ms network time                                            │
│                                                                         │
│     TOTAL BACKEND TIME: ~500-700ms                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Browser)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ 16. Response Handling                                                  │
│     - Parse JSON response                                              │
│     - Extract translation text                                         │
│     - Extract confidence score                                         │
│     - ~5-10ms                                                          │
│                                                                         │
│ 17. UI Update                                                          │
│     - Display translated text                                          │
│     - Update confidence indicator                                      │
│     - Show detected gestures                                           │
│     - Smooth transition (no flicker)                                   │
│     - ~10-20ms                                                         │
│                                                                         │
│     TOTAL END-TO-END TIME: ~860ms ✅ Under 1 second!                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Backend Responsibilities Explained

### 3.1 Frame Reception & Validation

**Purpose:** Securely receive and validate incoming video frames

**Code Example:**
```python
@video_bp.route('/process', methods=['POST'])
@require_auth  # Middleware for authentication
def process_video_frame(current_user):
    """Process a single video frame for ASL recognition"""
    
    # Get request data
    data = request.json
    frame_data = data.get('frame')
    
    # Validation
    if not frame_data:
        return jsonify({
            'success': False,
            'error': 'No frame data provided'
        }), 400
    
    # Size validation
    if len(frame_data) > 5_000_000:  # 5MB limit
        return jsonify({
            'success': False,
            'error': 'Frame size exceeds limit (5MB max)'
        }), 400
```

### 3.2 Authentication & Rate Limiting

**Purpose:** Ensure only authorized users can access the API and prevent abuse

**Code Example:**
```python
from functools import wraps

def rate_limit(requests_per_minute):
    """Rate limiting decorator based on user tier"""
    def decorator(f):
        @wraps(f)
        def decorated_function(current_user, *args, **kwargs):
            user_tier = current_user['subscriptionTier']
            
            # Get tier-specific limits
            limits = {
                'free': 10,
                'premium': 60,
                'enterprise': 300
            }
            
            limit = limits.get(user_tier, 10)
            
            # Check rate limit (using Redis or in-memory cache)
            if exceeded_rate_limit(current_user['uid'], limit):
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'limit': limit,
                    'resetTime': get_reset_time()
                }), 429
            
            return f(current_user, *args, **kwargs)
        return decorated_function
    return decorator

@video_bp.route('/process', methods=['POST'])
@require_auth
@rate_limit(requests_per_minute=10)
def process_video_frame(current_user):
    # Process frame...
```


### 3.3 Image Preprocessing

**Purpose:** Prepare raw image data for AI model consumption

**Code Example:**
```python
import cv2
import numpy as np
from PIL import Image
import io
import base64

def decode_and_preprocess_frame(base64_data: str) -> np.ndarray:
    """
    Decode base64 image and preprocess for model
    
    Returns:
        Preprocessed frame ready for MediaPipe/TensorFlow
    """
    # Remove data URL prefix if present
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    
    # Decode base64 to bytes
    image_data = base64.b64decode(base64_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to numpy array
    frame = np.array(image)
    
    # Resize to model input size (640x480)
    frame = cv2.resize(frame, (640, 480))
    
    # Convert BGR to RGB (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values (0-255 → 0-1)
    frame = frame.astype(np.float32) / 255.0
    
    return frame
```

### 3.4 MediaPipe Hand Detection

**Purpose:** Extract hand landmarks from video frames

**Code Example:**
```python
import mediapipe as mp

class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame: np.ndarray) -> dict:
        """
        Extract hand landmarks from frame
        
        Returns:
            {
                'hands_detected': int,
                'landmarks': list of landmark arrays,
                'handedness': list of 'Left' or 'Right'
            }
        """
        # Process frame
        results = self.hands.process(frame)
        
        if not results.multi_hand_landmarks:
            return {
                'hands_detected': 0,
                'landmarks': [],
                'handedness': []
            }
        
        # Extract landmarks for each hand
        landmarks_list = []
        handedness_list = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get 21 landmarks × 3 coordinates = 63 values per hand
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks_list.append(landmarks)
            
            # Get handedness (left or right)
            handedness = results.multi_handedness[idx].classification[0].label
            handedness_list.append(handedness)
        
        return {
            'hands_detected': len(landmarks_list),
            'landmarks': landmarks_list,
            'handedness': handedness_list
        }
```
### 3.5 Neural Network Inference

**Purpose:** Run the trained LSTM/Transformer model to recognize ASL gestures

**Code Example:**
```python
import tensorflow as tf
from tensorflow import keras

class ASLInferenceEngine:
    """Handle ASL gesture recognition inference"""
    
    def __init__(self, model_path: str):
        # Load model once at initialization (cached)
        self.model = keras.models.load_model(model_path)
        
        # Load gesture mapping
        with open('gesture_mapping.json', 'r') as f:
            self.idx_to_gesture = json.load(f)
    
    def predict(self, landmarks_sequence: np.ndarray) -> dict:
        """
        Predict ASL gesture from landmark sequence
        
        Args:
            landmarks_sequence: (30, 126) array of normalized landmarks
        
        Returns:
            {
                'gesture': str,
                'confidence': float,
                'top_k_predictions': list
            }
        """
        # Add batch dimension: (1, 30, 126)
        input_data = np.expand_dims(landmarks_sequence, axis=0)
        
        # Run inference
        predictions = self.model.predict(input_data, verbose=0)
        
        # Get top prediction
        gesture_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][gesture_idx])
        gesture_name = self.idx_to_gesture[str(gesture_idx)]
        
        # Get top-5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'gesture': self.idx_to_gesture[str(idx)],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        return {
            'gesture': gesture_name,
            'confidence': confidence,
            'top_k_predictions': top_5_predictions
        }

# Global instance (loaded once at startup)
inference_engine = ASLInferenceEngine('models/asl_model_v1.0.0.h5')
```

### 3.6 Translation History Storage

**Purpose:** Save translation results for user history and analytics

**Code Example:**
```python
from google.cloud import firestore
from datetime import datetime

def save_translation(
    user_id: str,
    translated_text: str,
    confidence_score: float,
    detected_gestures: list,
    session_id: str
) -> dict:
    """Save translation to Firestore"""
    
    db = firestore.Client()
    
    translation_data = {
        'translationId': str(uuid.uuid4()),
        'userId': user_id,
        'sessionId': session_id,
        'output': {
            'translatedText': translated_text,
            'finalText': translated_text
        },
        'model': {
            'version': 'v1.2.3',
            'confidenceScore': confidence_score
        },
        'timestamp': datetime.utcnow(),
        'userActions': {
            'wasEdited': False,
            'wasSaved': True
        }
    }
    
    # Save to Firestore (async, non-blocking)
    db.collection('users').document(user_id)\
      .collection('translations').add(translation_data)
    
    # Update user stats
    db.collection('users').document(user_id).update({
        'stats.totalTranslations': firestore.Increment(1),
        'stats.lastTranslationAt': datetime.utcnow()
    })
    
    return translation_data
```

### 3.7 Response Formatting & Error Handling

**Purpose:** Provide consistent, user-friendly API responses

**Code Example:**
```python
class APIResponse:
    """Standardized API response formatting"""
    
    @staticmethod
    def success(data: dict, status_code: int = 200):
        """Format success response"""
        return jsonify({
            'success': True,
            **data
        }), status_code
    
    @staticmethod
    def error(message: str, error_code: str = None, status_code: int = 400):
        """Format error response"""
        response = {
            'success': False,
            'error': message
        }
        
        if error_code:
            response['errorCode'] = error_code
        
        return jsonify(response), status_code

# Usage in route
@video_bp.route('/process', methods=['POST'])
def process_frame():
    try:
        # Process frame
        result = process_video_frame(frame_data)
        
        return APIResponse.success({
            'translation': result['translation'],
            'performance': result['performance']
        })
        
    except NoHandsDetectedError:
        return APIResponse.error(
            message='No hands detected in frame. Please position your hands in view.',
            error_code='NO_HANDS_DETECTED',
            status_code=400
        )
    
    except ModelInferenceError as e:
        return APIResponse.error(
            message='Translation processing failed. Please try again.',
            error_code='INFERENCE_FAILED',
            status_code=500
        )
    
    except Exception as e:
        # Log error for debugging
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        
        return APIResponse.error(
            message='An unexpected error occurred.',
            error_code='INTERNAL_ERROR',
            status_code=500
        )
```


## 4. Why Backend Processing is Essential

### 4.1 Frontend vs Backend Capabilities

| Capability | Frontend (Browser) | Backend (Flask Server) |
|------------|-------------------|------------------------|
| **MediaPipe** | ❌ Not available in JavaScript | ✅ Full Python library |
| **TensorFlow Model** | ⚠️ TensorFlow.js (slower, limited) | ✅ Native TensorFlow (fast, full GPU) |
| **Model Size** | ❌ 50MB+ download per user | ✅ Loaded once, cached in memory |
| **GPU Acceleration** | ⚠️ Limited WebGL support | ✅ Full NVIDIA CUDA support |
| **Processing Power** | ⚠️ Limited by user's device | ✅ Powerful server hardware |
| **Security** | ❌ Code visible to users | ✅ Protected server-side |
| **Rate Limiting** | ❌ Can be bypassed | ✅ Server-enforced |
| **Database Access** | ❌ Security risk | ✅ Secure credentials |
| **API Keys** | ❌ Exposed to public | ✅ Protected |
| **Consistency** | ⚠️ Varies by device/browser | ✅ Same results for everyone |

---


## 5. Comparison: Object Detection Web App

### 5.1 Overview of the Medium Article Approach

The article "[Build a Computer Vision Web App — Flask, OpenCV, and MongoDB](https://medium.com/better-programming/build-a-computer-vision-webapp-flask-opencv-and-mongodb-62a52d38738a)" by Zaheeda Chauke describes building an object detection web application using:

- **Flask** (backend framework)
- **OpenCV** (computer vision library)
- **cvlib** (high-level CV wrapper)
- **MongoDB** (NoSQL database for results storage)

**Purpose:** Upload images/videos → Detect objects → Save results to database

---

### 5.2 Their Backend Functions Breakdown

#### **Function 1: Image Upload Route**

```python
@app.route('/', methods=["POST"])
def uploadFile():
    """Handle file upload from user"""
    
    # Get uploaded file
    _img = request.files['file-uploaded']
    
    # Validate file type
    if _img and allowed_file(_img.filename):
        # Save to uploads folder
        filename = secure_filename(_img.filename)
        _img.save(os.path.join(UPLOAD_FOLDER, filename))
        
        # Store path in session
        session['uploaded_img_file_path'] = os.path.join(
            UPLOAD_FOLDER, 
            filename
        )
        
        # Return success
        return render_template('index.html', success=True)
```

**Purpose:** 
- Receive uploaded image/video from frontend
- Validate file type (images: jpg, png; videos: mp4)
- Save to server filesystem
- Store file path in session for later use

**Key Differences from SharedHands:**
- They save the entire file to disk
- We process frames in memory and delete immediately (privacy REQ029)
- They use session storage for file paths
- We use database references

---

#### **Function 2: Display Uploaded File**

```python
@app.route('/show_file')
def showFile():
    """Display the uploaded file to user"""
    
    # Get file path from session
    uploaded_img_path = session.get('uploaded_img_file_path', None)
    
    # Determine if image or video
    is_image = uploaded_img_path.endswith(('.jpg', '.png', '.jpeg'))
    
    # Render template with file
    return render_template(
        'show_file.html',
        user_image=uploaded_img_path,
        is_image=is_image,
        is_show_button=True  # Show "Detect Objects" button
    )
```

**Purpose:**
- Retrieve uploaded file path
- Determine file type
- Display file to user with "Detect Objects" button

**Key Differences from SharedHands:**
- They display uploaded files from disk
- We stream video in real-time from camera
- They process on-demand (button click)
- We process continuously (2-5 FPS)

---

#### **Function 4: Video Object Detection**

```python
def detect_and_draw_box_video(video_filepath, confidence=0.5, model='yolov3'):
    """
    Detect objects in video frames
    
    Args:
        video_filepath: Path to video file
        confidence: Detection confidence threshold
        model: YOLO model to use
    
    Returns:
        output_video_path: Path to annotated video
        response: Aggregated detection results
    """
    import cv2
    import cvlib as cv
    from cvlib.object_detection import draw_bbox
    
    # Open video file
    cap = cv2.VideoCapture(video_filepath)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_path = os.path.join(OUTPUT_FOLDER, 'output_video.avi')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    all_detections = []
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Flip frame (may be upside down)
        frame = cv2.flip(frame, 1)
        
        # Detect objects in frame
        bbox, label, conf = cv.detect_common_objects(
            frame,
            confidence=confidence,
            model=model
        )
        
        # Draw boxes
        output_frame = draw_bbox(frame, bbox, label, conf)
        
        # Write to output video
        out.write(output_frame)
        
        # Collect detections
        for lbl, cnf in zip(label, conf):
            all_detections.append({
                'label': lbl,
                'confidence': float(cnf)
            })
        
        # Display frame (optional, for debugging)
        cv2.imshow('Video Detection', output_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Aggregate results
    response = {
        'total_frames_processed': len(all_detections),
        'unique_objects': list(set([d['label'] for d in all_detections])),
        'all_detections': all_detections
    }
    
    # Save results
    save_response_json(response, 'video_detection_result.json')
    add_data_to_mongodb(response)
    
    return out_path, response
```

**Purpose:**
- Read video file frame by frame
- Detect objects in each frame
- Create annotated output video
- Aggregate all detections


### 6.4 Key Architectural Insights

#### **What We Learn from Object Detection App:**

1. **Session Storage Pattern**
   - Using Flask sessions to store file paths between requests
   - SharedHands could use this for maintaining conversation context

2. **File Validation**
   - Important to validate file types server-side
   - SharedHands validates frame format (JPEG/PNG)

3. **Response Structure**
   - Save results both locally (JSON) and in database
   - SharedHands only saves to database (privacy)

4. **Video Processing Loop**
   - Use `cv2.VideoCapture()` for frame-by-frame processing
   - SharedHands processes live stream instead of files

5. **MongoDB Integration**
   - Simple pattern: create client → get collection → insert data
   - Firestore has similar pattern but different API

---



### 7.3 Combined Best Practices

```python
# GOOD: Complete backend route pattern
@app.route('/process', methods=['POST'])
@require_auth              # Authentication middleware
@rate_limit(tier_based)    # Rate limiting
def process_request(current_user):
    try:
        # 1. Validate input
        data = request.json
        if not validate_input(data):
            return APIResponse.error("Invalid input", 400)
        
        # 2. Process (with timing)
        start_time = time.time()
        result = heavy_processing(data)
        processing_time = time.time() - start_time
        
        # 3. Save to database (async)
        save_result_async(result, current_user['uid'])
        
        # 4. Return response
        return APIResponse.success({
            'result': result,
            'processing_time_ms': int(processing_time * 1000)
        })
        
    except ValidationError as e:
        return APIResponse.error(str(e), 400)
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
        return APIResponse.error("Processing failed", 500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return APIResponse.error("Internal error", 500)
```

---

### 7.4 Performance Optimization Strategies

**From Both Applications:**

1. **Cache Heavy Resources**
   ```python
   # Load model once at startup, not per request
   model = load_model('model.h5')  # Global variable
   ```

2. **Async Database Writes**
   ```python
   # Don't block response on database write
   threading.Thread(target=save_to_db, args=(data,)).start()
   ```

3. **Input Validation Early**
   ```python
   # Fail fast if input is invalid
   if not valid_input(data):
       return error_response()
   # Don't waste time processing invalid data
   ```

4. **Batch Processing When Possible**
   ```python
   # Process multiple frames together (if not real-time)
   results = model.predict(frames_batch)
   ```

5. **Use Appropriate Data Structures**
   ```python
   # NumPy for numerical data (faster than lists)
   frame = np.array(image)
   ```


**References:**
- Medium Article: [Build a Computer Vision Web App](https://medium.com/better-programming/build-a-computer-vision-webapp-flask-opencv-and-mongodb-62a52d38738a)
- SharedHands Project Documentation
- Flask Documentation: https://flask.palletsprojects.com/
- MediaPipe Documentation: https://google.github.io/mediapipe/
- TensorFlow Documentation: https://www.tensorflow.org/
