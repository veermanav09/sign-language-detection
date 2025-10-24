from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import json
from sign_recognition_rule_based import RuleBasedSignRecognizer

app = Flask(__name__)
CORS(app)

# Global variables
recognizer = None
camera = None
is_processing = False
current_sign = "No Sign"
confidence = 0.0
frame_buffer = None

def initialize_recognizer():
    """Initialize the sign recognizer."""
    global recognizer
    try:
        recognizer = RuleBasedSignRecognizer()
        print("Sign recognizer initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize recognizer: {e}")
        return False

def initialize_camera():
    """Initialize the camera."""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not camera.isOpened():
            print("Failed to open camera")
            return False
            
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return False

def process_frames():
    """Process camera frames in a separate thread."""
    global is_processing, frame_buffer, current_sign, confidence
    
    while is_processing:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        try:
            # Process frame for sign recognition
            if recognizer:
                processed_frame, sign, conf = recognizer.process_frame(frame)
                current_sign = sign
                confidence = conf
                
                # Encode frame for web display
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_buffer = buffer.tobytes()
            else:
                # Just encode the original frame if recognizer is not available
                _, buffer = cv2.imencode('.jpg', frame)
                frame_buffer = buffer.tobytes()
                
        except Exception as e:
            print(f"Frame processing error: {e}")
            time.sleep(0.1)
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            if frame_buffer is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
            else:
                # Send a placeholder frame
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current recognition status."""
    return jsonify({
        'sign': current_sign,
        'confidence': round(confidence, 3),
        'is_processing': is_processing,
        'camera_ready': camera is not None and camera.isOpened() if camera else False
    })

@app.route('/api/start', methods=['POST'])
def start_recognition():
    """Start sign recognition."""
    global is_processing
    
    if not recognizer:
        if not initialize_recognizer():
            return jsonify({'success': False, 'error': 'Failed to initialize recognizer'}), 500
    
    if not camera:
        if not initialize_camera():
            return jsonify({'success': False, 'error': 'Failed to initialize camera'}), 500
    
    if not is_processing:
        is_processing = True
        # Start processing thread
        threading.Thread(target=process_frames, daemon=True).start()
        return jsonify({'success': True, 'message': 'Recognition started'})
    else:
        return jsonify({'success': False, 'error': 'Already processing'})

@app.route('/api/stop', methods=['POST'])
def stop_recognition():
    """Stop sign recognition."""
    global is_processing
    
    if is_processing:
        is_processing = False
        return jsonify({'success': True, 'message': 'Recognition stopped'})
    else:
        return jsonify({'success': False, 'error': 'Not currently processing'})

@app.route('/api/speak', methods=['POST'])
def speak_sign():
    """Manually trigger speech for a sign."""
    data = request.get_json()
    sign = data.get('sign', '')
    
    if recognizer and sign:
        try:
            recognizer.speak_sign_async(sign)
            return jsonify({'success': True, 'message': f'Speaking: {sign}'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Invalid sign or recognizer not ready'}), 400

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update recognition settings."""
    if request.method == 'GET':
        if recognizer:
            return jsonify({
                'confidence_threshold': recognizer.confidence_threshold,
                'stable_frames_threshold': recognizer.stable_frames_threshold
            })
        else:
            return jsonify({'error': 'Recognizer not initialized'}), 500
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if recognizer:
            try:
                if 'confidence_threshold' in data:
                    recognizer.confidence_threshold = float(data['confidence_threshold'])
                if 'stable_frames_threshold' in data:
                    recognizer.stable_frames_threshold = int(data['stable_frames_threshold'])
                
                return jsonify({'success': True, 'message': 'Settings updated'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        else:
            return jsonify({'success': False, 'error': 'Recognizer not initialized'}), 500

@app.route('/api/signs')
def get_available_signs():
    """Get list of available signs."""
    if recognizer:
        return jsonify({
            'signs': recognizer.sign_mapping,
            'total_count': len(recognizer.sign_mapping)
        })
    else:
        return jsonify({'error': 'Recognizer not initialized'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def cleanup():
    """Cleanup resources."""
    global is_processing, camera, recognizer
    
    is_processing = False
    
    if camera:
        camera.release()
    
    if recognizer:
        recognizer.hand_tracker.release()

# Register cleanup on app shutdown
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    print("Starting Sign Language Recognition Web App...")
    print("Initializing components...")
    
    # Initialize recognizer
    if initialize_recognizer():
        print("✓ Sign recognizer ready")
    else:
        print("✗ Sign recognizer failed to initialize")
    
    # Initialize camera
    if initialize_camera():
        print("✓ Camera ready")
    else:
        print("✗ Camera failed to initialize")
    
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
        print("Server stopped.")
