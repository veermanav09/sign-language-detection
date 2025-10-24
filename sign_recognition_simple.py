import cv2
import numpy as np
import pyttsx3
import time
import threading
from typing import List, Tuple, Optional
from hand_tracking_opencv import HandTrackerOpenCV

class SimpleSignRecognizer:
    """
    Simplified sign language recognition using OpenCV.
    Compatible with Python 3.13 and ARM64 systems.
    """
    
    def __init__(self):
        """Initialize the simplified sign recognizer."""
        self.hand_tracker = HandTrackerOpenCV()
        self.engine = pyttsx3.init()
        
        # Configure text-to-speech engine
        self._configure_tts()
        
        # Simple sign mapping based on hand position and size
        self.sign_mapping = {
            'A': 'A - Fist with thumb on side',
            'B': 'B - Flat hand, fingers together',
            'C': 'C - Curved hand, like holding a ball',
            'D': 'D - Index finger pointing up',
            'E': 'E - Fist with fingers bent',
            'F': 'F - Index and thumb touching, other fingers up',
            'G': 'G - Index and thumb pointing, other fingers down',
            'H': 'H - Index and middle finger pointing up',
            'I': 'I - Pinky finger pointing up',
            'J': 'J - Pinky finger moving in J shape',
            'K': 'K - Index and middle finger pointing up, thumb between',
            'L': 'L - Thumb and index finger forming L',
            'M': 'M - Three fingers down, thumb and pinky up',
            'N': 'N - Two fingers down, three up',
            'O': 'O - Fingers curved, like holding a ball',
            'P': 'P - Index finger pointing down',
            'Q': 'Q - Index finger pointing down, thumb up',
            'R': 'R - Index and middle finger crossed',
            'S': 'S - Fist',
            'T': 'T - Index finger pointing up, thumb across palm',
            'U': 'U - Index and middle finger pointing up',
            'V': 'V - Index and middle finger pointing up, spread',
            'W': 'W - Three fingers pointing up',
            'X': 'X - Index finger pointing up, bent',
            'Y': 'Y - Thumb and pinky pointing up',
            'Z': 'Z - Index finger moving in Z shape'
        }
        
        # Recognition state
        self.last_recognized_sign = None
        self.recognition_confidence = 0.0
        self.is_speaking = False
        self.speech_lock = threading.Lock()
        self.last_spoken_at = 0.0
        self.speak_cooldown_secs = 2.0
        
        # Recognition parameters (tunable)
        # Slightly lower thresholds to improve recall, but keep stability checks
        self.confidence_threshold = 0.5
        self.stable_frames_threshold = 4
        self.max_history = 20
        self.last_signs = []
        
        # Simple gesture recognition rules (kept for future tuning)
        self.gesture_rules = {
            'fist': ['S', 'A', 'E'],
            'open_palm': ['B', '5'],
            'pointing': ['D', '1'],
            'peace': ['V', '2'],
            'thumbs_up': ['A', 'Good'],
            'thumbs_down': ['Bad'],
            'okay': ['F', 'O', 'Okay']
        }
    
    def _configure_tts(self):
        """Configure text-to-speech engine settings."""
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Set voice properties
            if voices:
                # Try to use a female voice if available
                # Prefer English voices with clear names, fallback to first
                preferred = None
                for voice in voices:
                    name = voice.name.lower()
                    if ('english' in name or 'en_' in name or 'en-' in name) and ('female' in name or 'us' in name or 'uk' in name):
                        preferred = voice
                        break
                if not preferred:
                    for voice in voices:
                        if 'female' in voice.name.lower():
                            preferred = voice
                            break
                self.engine.setProperty('voice', (preferred or voices[0]).id)
            
            # Set speech rate, volume and try clearer voice
            # Slightly slower rate improves intelligibility
            self.engine.setProperty('rate', 140)  # Words per minute
            # Keep volume high but avoid clipping
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            
        except Exception as e:
            print(f"Warning: Could not configure TTS engine: {e}")
    
    def _classify_gesture(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify gesture based on hand features.
        
        Args:
            features: Hand features array
            
        Returns:
            Tuple of (predicted_sign, confidence)
        """
        if features is None or len(features) < 47:
            return "No Sign", 0.0
        
        try:
            # Extract key features
            width_norm = features[42]  # Width
            height_norm = features[43]  # Height
            aspect_ratio = features[44]  # Aspect ratio
            center_x = features[45]  # Center X
            center_y = features[46]  # Center Y
            
            # Simple gesture classification based on hand shape
            confidence = 0.0
            predicted_sign = "Unknown"
            
            # Rule 1: Fist detection (small aspect ratio, centered)
            if aspect_ratio < 0.8 and 0.3 < center_x < 0.7 and 0.3 < center_y < 0.7:
                predicted_sign = "S"  # Fist
                confidence = 0.8
            
            # Rule 2: Open palm (moderate to large aspect ratio)
            elif aspect_ratio > 1.1:
                predicted_sign = "B"  # Open palm
                confidence = 0.65
            
            # Rule 3: Pointing gesture (high aspect ratio, off-center)
            elif aspect_ratio > 1.5 and (center_x < 0.3 or center_x > 0.7):
                predicted_sign = "D"  # Pointing
                confidence = 0.6
            
            # Rule 4: Peace sign (near-square aspect ratio, reasonably centered)
            elif 0.85 < aspect_ratio < 1.15 and 0.35 < center_x < 0.65:
                predicted_sign = "V"  # Peace
                confidence = 0.65
            
            # Rule 5: Thumbs up (low aspect ratio, top of frame)
            elif aspect_ratio < 0.7 and center_y < 0.4:
                predicted_sign = "Good"
                confidence = 0.6
            
            # Rule 6: Thumbs down (low aspect ratio, bottom of frame)
            elif aspect_ratio < 0.7 and center_y > 0.6:
                predicted_sign = "Bad"
                confidence = 0.6
            
            # Rule 7: OK sign (medium aspect ratio, specific position)
            elif 0.8 < aspect_ratio < 1.05 and 0.35 < center_x < 0.65 and 0.35 < center_y < 0.65:
                predicted_sign = "O"  # OK
                confidence = 0.55
            
            # Add slight noise clamp
            if confidence > 0:
                confidence = float(max(0.0, min(1.0, confidence)))
            
            return predicted_sign, confidence
            
        except Exception as e:
            print(f"Gesture classification error: {e}")
            return "Error", 0.0
    
    def speak_sign(self, sign: str):
        """
        Convert sign to speech output.
        
        Args:
            sign: Sign to speak
        """
        with self.speech_lock:
            if self.is_speaking:
                return
            
            # Debounce speaking
            now = time.time()
            if now - self.last_spoken_at < self.speak_cooldown_secs:
                return
            self.last_spoken_at = now
            
            self.is_speaking = True
            
            try:
                # Clean up sign text for speech
                if sign in self.sign_mapping:
                    speech_text = self.sign_mapping[sign]
                elif sign.isalpha() and len(sign) == 1:
                    speech_text = f"The letter {sign}"
                elif sign.isdigit():
                    speech_text = f"The number {sign}"
                else:
                    speech_text = sign
                
                # Speak the text
                self.engine.say(speech_text)
                self.engine.runAndWait()
                
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.is_speaking = False
    
    def speak_sign_async(self, sign: str):
        """Speak sign asynchronously to avoid blocking video processing."""
        threading.Thread(target=self.speak_sign, args=(sign,), daemon=True).start()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Process a video frame for sign recognition.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, recognized_sign, confidence)
        """
        # Detect hands and extract features
        processed_frame, hand_features_list = self.hand_tracker.detect_hands(frame)
        
        recognized_sign = "No Sign"
        confidence = 0.0
        
        if hand_features_list:
            # Use the first detected hand
            features = hand_features_list[0]
            
            # Classify gesture
            predicted_sign, pred_confidence = self._classify_gesture(features)
            
            if predicted_sign != "Unknown" and pred_confidence >= self.confidence_threshold:
                # Update recognition history
                self.last_signs.append(predicted_sign)
                if len(self.last_signs) > self.max_history:
                    self.last_signs.pop(0)
                
                # Check for stable recognition
                if len(self.last_signs) >= self.stable_frames_threshold:
                    from collections import Counter
                    # Use a slightly larger window for stability voting
                    window = max(self.stable_frames_threshold, 6)
                    counts = Counter(self.last_signs[-window:])
                    most_common = counts.most_common(1)[0]
                    stable_sign, stable_count = most_common
                    
                    # Require majority in the window to reduce flicker
                    if stable_count >= (window // 2 + 1):
                        if stable_sign != self.last_recognized_sign:
                            self.last_recognized_sign = stable_sign
                            self.speak_sign_async(stable_sign)
                        recognized_sign = stable_sign
                        confidence = pred_confidence
                
                # Display prediction info
                cv2.putText(processed_frame, f"Predicted: {predicted_sign}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confidence: {pred_confidence:.2f}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current recognition
        cv2.putText(processed_frame, f"Recognized: {recognized_sign}", 
                   (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add instructions
        cv2.putText(processed_frame, "Make hand gestures to see recognition | c: calibrate | r: ROI toggle", 
                   (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        return processed_frame, recognized_sign, confidence
    
    def run_realtime(self, camera_index: int = 0):
        """
        Run real-time sign recognition from camera.
        
        Args:
            camera_index: Camera device index
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Simple Sign Recognition Started!")
        print("This version uses OpenCV instead of MediaPipe")
        print("Make hand gestures in front of the camera")
        print("Press 'q' to quit, 's' to speak current sign, 'c' to calibrate, 'r' ROI toggle")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, sign, conf = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Simple Sign Recognition', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if sign != "No Sign":
                        self.speak_sign_async(sign)
                        print(f"Speaking: {sign}")
                    else:
                        print("No sign detected to speak")
                elif key == ord('c'):
                    # Calibrate skin thresholds and reset bg
                    self.hand_tracker.calibrate(frame)
                    print("Calibrated skin thresholds and background reset.")
                elif key == ord('r'):
                    self.hand_tracker.set_center_roi(not self.hand_tracker.use_center_roi)
                    state = 'ON' if self.hand_tracker.use_center_roi else 'OFF'
                    print(f"Center ROI: {state}")
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()
    
    def get_available_signs(self):
        """Get list of available signs."""
        return self.sign_mapping

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Sign Language Recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = SimpleSignRecognizer()
    
    # Show available signs
    print("Available signs:")
    for sign, description in recognizer.get_available_signs().items():
        print(f"  {sign}: {description}")
    print()
    
    # Run real-time recognition
    recognizer.run_realtime(camera_index=args.camera)
