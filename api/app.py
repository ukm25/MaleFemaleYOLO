#!/usr/bin/env python3
"""
Flask API Server for Gender and Age Detection
Simple web API that receives images and returns detection results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
from yolo_detector import YOLODetector
from gender_classifier import GenderClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize models
print("🎯 Initializing models...")
detector = YOLODetector(model_path="yolov8n.pt")  # Use nano for speed
gender_classifier = GenderClassifier(model_name='OpenFace')  # Use OpenFace for lowest memory
print("✅ Models loaded successfully!")

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def process_image(image):
    """Process image with YOLO + DeepFace"""
    try:
        start_time = time.time()
        
        # Detect persons
        detections = detector.detect_persons(image)
        
        results = []
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Crop person from image
            person_crop = image[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if person_crop.shape[0] < 50 or person_crop.shape[1] < 50:
                continue
            
            # Analyze with DeepFace
            analysis = gender_classifier.analyze_face(person_crop)
            
            # Add bounding box info
            analysis['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
            analysis['detection_confidence'] = float(confidence)
            
            results.append(analysis)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'detections': results,
            'processing_time': round(processing_time, 3),
            'timestamp': time.time()
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }

@app.route('/api/detect', methods=['POST'])
def detect_gender_age():
    """API endpoint for gender and age detection"""
    try:
        # Get image from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image format'
            }), 400
        
        # Process image
        result = process_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def index():
    """Simple index page"""
    return """
    <h1>Gender & Age Detection API</h1>
    <p>API is running successfully!</p>
    <p>Endpoints:</p>
    <ul>
        <li>POST /api/detect - Detect gender and age</li>
        <li>GET /api/health - Health check</li>
    </ul>
    """

# For Vercel serverless functions
def handler(request):
    return app(request.environ, start_response)

# For local development
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("🚀 Starting Flask API server...")
    print(f"📡 API will be available at: http://0.0.0.0:{port}")
    print("🎯 Ready to receive images for gender and age detection!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
