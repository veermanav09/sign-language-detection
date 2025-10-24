import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyttsx3
import time
import threading
from typing import List, Tuple, Optional
from hand_tracking import HandTracker

class SignRecognizer:
    """
    Real-time sign language recognition with audio output.
    Combines hand tracking, sign classification, and text-to-speech.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the sign recognizer.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.hand_tracker = HandTracker()
        self.engine = pyttsx3.init()
        
        # Configure text-to-speech engine
        self._configure_tts()
        
        # Sign mapping (ASL alphabet and common words)
        self.sign_mapping = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
            26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
            36: 'Hello', 37: 'Thank You', 38: 'Yes', 39: 'No', 40: 'Please', 41: 'Sorry', 42: 'Good', 43: 'Bad'
        }
        
        # Load or create model
        if model_path and tf.io.gfile.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = self._create_model()
            print("Created new model (training required)")
        
        # Recognition state
        self.last_recognized_sign = None
        self.recognition_confidence = 0.0
        self.is_speaking = False
        self.speech_lock = threading.Lock()
        
        # Recognition parameters
        # Lower threshold for improved sensitivity; stability still enforced
        self.confidence_threshold = 0.6
        self.stable_frames_threshold = 4
        self.stable_frames_count = 0
        self.last_signs = []
    
    def _configure_tts(self):
        """Configure text-to-speech engine settings."""
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Set voice properties
            if voices:
                # Prefer clear English voices; fall back to first
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
            
            # Set speech rate and volume for clarity
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', 1.0)
            
        except Exception as e:
            print(f"Warning: Could not configure TTS engine: {e}")
    
    def _create_model(self) -> tf.keras.Model:
        """
        Create a neural network model for sign recognition.
        
        Returns:
            Compiled Keras model
        """
        # Input shape: 42 hand landmarks + angles
        input_shape = (47,)  # 42 coordinates + 5 angles
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # Normalization
            tf.keras.layers.BatchNormalization(),
            
            # Hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(len(self.sign_mapping), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_sign(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict sign from hand features.
        
        Args:
            features: Hand features array
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if features is None or len(features) == 0:
            return -1, 0.0
        
        # Ensure features are the right shape
        if len(features) != 47:
            # Pad or truncate to match expected input
            if len(features) < 47:
                features = np.pad(features, (0, 47 - len(features)), 'constant')
            else:
                features = features[:47]
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Make prediction
        try:
            predictions = self.model.predict(features, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            return predicted_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return -1, 0.0
    
    def speak_sign(self, sign: str):
        """
        Convert sign to speech output.
        
        Args:
            sign: Sign to speak
        """
        with self.speech_lock:
            if self.is_speaking:
                return
            
            self.is_speaking = True
            
            try:
                # Clean up sign text for speech
                if sign.isalpha() and len(sign) == 1:
                    # Single letter - spell it out
                    speech_text = f"The letter {sign}"
                elif sign.isdigit():
                    # Number
                    speech_text = f"The number {sign}"
                else:
                    # Word
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
            
            # Predict sign
            predicted_class, pred_confidence = self.predict_sign(features)
            
            if predicted_class != -1 and pred_confidence > self.confidence_threshold:
                # Update recognition state
                self.last_signs.append(predicted_class)
                if len(self.last_signs) > 10:
                    self.last_signs.pop(0)
                
                # Check for stable recognition
                if len(self.last_signs) >= self.stable_frames_threshold:
                    # Get most common sign in recent frames
                    from collections import Counter
                    most_common = Counter(self.last_signs).most_common(1)[0]
                    
                    if most_common[1] >= self.stable_frames_threshold:
                        stable_sign = most_common[0]
                        sign_text = self.sign_mapping.get(stable_sign, "Unknown")
                        
                        # Only speak if it's a new sign
                        if sign_text != self.last_recognized_sign:
                            self.last_recognized_sign = sign_text
                            self.speak_sign_async(sign_text)
                        
                        recognized_sign = sign_text
                        confidence = pred_confidence
                
                # Display prediction info
                cv2.putText(processed_frame, f"Predicted: {self.sign_mapping.get(predicted_class, 'Unknown')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confidence: {pred_confidence:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current recognition
        cv2.putText(processed_frame, f"Recognized: {recognized_sign}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
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
        
        print("Real-time sign recognition started. Press 'q' to quit.")
        print("Make sure your hand is clearly visible in the camera.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, sign, conf = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Sign Language Recognition', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if tf.io.gfile.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time sign language recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    parser.add_argument('--output', choices=['audio', 'text'], default='audio', 
                       help='Output type: audio or text only')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = SignRecognizer(model_path=args.model)
    
    # Disable audio if not requested
    if args.output == 'text':
        recognizer.speak_sign = lambda x: None
    
    # Run real-time recognition
    recognizer.run_realtime(camera_index=args.camera)
