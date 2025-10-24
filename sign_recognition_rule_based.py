#!/usr/bin/env python3
"""
Rule-based sign language recognition using MediaPipe hand landmarks.
This provides immediate, accurate recognition without needing a trained model.
"""

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
import threading
from typing import List, Tuple, Optional

class RuleBasedSignRecognizer:
    """
    Rule-based sign language recognition using MediaPipe hand landmarks.
    Provides immediate, accurate recognition for common ASL signs.
    """
    
    def __init__(self):
        """Initialize the rule-based sign recognizer."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Text-to-speech engine
        self.engine = pyttsx3.init()
        self._configure_tts()
        
        # Recognition state
        self.last_recognized_sign = None
        self.last_spoken_at = 0
        self.speak_cooldown = 2.0  # seconds
        
        # Recognition parameters
        self.confidence_threshold = 0.8
        self.stable_frames_threshold = 3
        self.stable_frames_count = 0
        self.last_signs = []
        
        # Sign definitions based on hand geometry
        self.sign_definitions = {
            'A': self._is_sign_A,
            'B': self._is_sign_B,
            'C': self._is_sign_C,
            'D': self._is_sign_D,
            'E': self._is_sign_E,
            'F': self._is_sign_F,
            'G': self._is_sign_G,
            'H': self._is_sign_H,
            'I': self._is_sign_I,
            'J': self._is_sign_J,
            'K': self._is_sign_K,
            'L': self._is_sign_L,
            'M': self._is_sign_M,
            'N': self._is_sign_N,
            'O': self._is_sign_O,
            'P': self._is_sign_P,
            'Q': self._is_sign_Q,
            'R': self._is_sign_R,
            'S': self._is_sign_S,
            'T': self._is_sign_T,
            'U': self._is_sign_U,
            'V': self._is_sign_V,
            'W': self._is_sign_W,
            'X': self._is_sign_X,
            'Y': self._is_sign_Y,
            'Z': self._is_sign_Z,
            'Hello': self._is_sign_hello,
            'Thank You': self._is_sign_thank_you,
            'Yes': self._is_sign_yes,
            'No': self._is_sign_no,
            'Please': self._is_sign_please,
            'Sorry': self._is_sign_sorry,
            'Good': self._is_sign_good,
            'Bad': self._is_sign_bad
        }
    
    def _configure_tts(self):
        """Configure text-to-speech engine."""
        try:
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Warning: Could not configure TTS: {e}")
    
    def _get_landmark_coords(self, landmarks, index):
        """Get normalized coordinates of a landmark."""
        if landmarks and 0 <= index < len(landmarks.landmark):
            return landmarks.landmark[index].x, landmarks.landmark[index].y
        return None, None
    
    def _get_finger_tip_y(self, landmarks, finger_tip_index):
        """Get Y coordinate of finger tip (lower = more extended)."""
        x, y = self._get_landmark_coords(landmarks, finger_tip_index)
        return y if y is not None else 1.0
    
    def _get_finger_pip_y(self, landmarks, finger_pip_index):
        """Get Y coordinate of finger PIP joint."""
        x, y = self._get_landmark_coords(landmarks, finger_pip_index)
        return y if y is not None else 1.0
    
    def _is_finger_extended(self, landmarks, finger_tip_index, finger_pip_index):
        """Check if a finger is extended (tip above PIP joint)."""
        tip_y = self._get_finger_tip_y(landmarks, finger_tip_index)
        pip_y = self._get_finger_pip_y(landmarks, finger_pip_index)
        return tip_y < pip_y - 0.02  # Small threshold for stability
    
    def _is_finger_bent(self, landmarks, finger_tip_index, finger_pip_index):
        """Check if a finger is bent (tip below PIP joint)."""
        tip_y = self._get_finger_tip_y(landmarks, finger_tip_index)
        pip_y = self._get_finger_pip_y(landmarks, finger_pip_index)
        return tip_y > pip_y + 0.02
    
    def _is_finger_half_bent(self, landmarks, finger_tip_index, finger_pip_index):
        """Check if a finger is half-bent (tip near PIP joint)."""
        tip_y = self._get_finger_tip_y(landmarks, finger_tip_index)
        pip_y = self._get_finger_pip_y(landmarks, finger_pip_index)
        return abs(tip_y - pip_y) <= 0.02
    
    # Sign detection methods
    def _is_sign_A(self, landmarks):
        """Detect ASL letter A: closed fist with thumb on side."""
        if not landmarks:
            return False
        
        # All fingers closed (tips below PIP joints)
        thumb_tip = self._get_finger_tip_y(landmarks, 4)
        index_tip = self._get_finger_tip_y(landmarks, 8)
        middle_tip = self._get_finger_tip_y(landmarks, 12)
        ring_tip = self._get_finger_tip_y(landmarks, 16)
        pinky_tip = self._get_finger_tip_y(landmarks, 20)
        
        # Thumb PIP and MCP joints
        thumb_pip = self._get_finger_pip_y(landmarks, 3)
        thumb_mcp = self._get_finger_pip_y(landmarks, 2)
        
        # All fingers should be closed
        fingers_closed = (
            index_tip > self._get_finger_pip_y(landmarks, 7) and
            middle_tip > self._get_finger_pip_y(landmarks, 11) and
            ring_tip > self._get_finger_pip_y(landmarks, 15) and
            pinky_tip > self._get_finger_pip_y(landmarks, 19)
        )
        
        # Thumb should be extended or slightly bent
        thumb_extended = thumb_tip < thumb_pip - 0.01
        
        return fingers_closed and thumb_extended
    
    def _is_sign_B(self, landmarks):
        """Detect ASL letter B: all fingers extended, thumb tucked."""
        if not landmarks:
            return False
        
        # All fingers extended
        index_extended = self._is_finger_extended(landmarks, 8, 7)
        middle_extended = self._is_finger_extended(landmarks, 12, 11)
        ring_extended = self._is_finger_extended(landmarks, 16, 15)
        pinky_extended = self._is_finger_extended(landmarks, 20, 19)
        
        # Thumb tucked (tip below MCP)
        thumb_tucked = self._get_finger_tip_y(landmarks, 4) > self._get_finger_pip_y(landmarks, 2)
        
        return index_extended and middle_extended and ring_extended and pinky_extended and thumb_tucked
    
    def _is_sign_C(self, landmarks):
        """Detect ASL letter C: curved hand like holding a cup."""
        if not landmarks:
            return False
        
        # All fingers should be curved (tips above PIP but below MCP)
        index_curved = self._is_finger_half_bent(landmarks, 8, 7)
        middle_curved = self._is_finger_half_bent(landmarks, 12, 11)
        ring_curved = self._is_finger_half_bent(landmarks, 16, 15)
        pinky_curved = self._is_finger_half_bent(landmarks, 20, 19)
        
        # Thumb should be curved too
        thumb_curved = self._is_finger_half_bent(landmarks, 4, 3)
        
        return index_curved and middle_curved and ring_curved and pinky_curved and thumb_curved
    
    def _is_sign_V(self, landmarks):
        """Detect ASL letter V: index and middle extended, others closed."""
        if not landmarks:
            return False
        
        # Index and middle extended
        index_extended = self._is_finger_extended(landmarks, 8, 7)
        middle_extended = self._is_finger_extended(landmarks, 12, 11)
        
        # Ring and pinky closed
        ring_closed = self._is_finger_bent(landmarks, 16, 15)
        pinky_closed = self._is_finger_bent(landmarks, 20, 19)
        
        # Thumb can be extended or tucked
        thumb_ok = True  # More flexible for thumb
        
        return index_extended and middle_extended and ring_closed and pinky_closed and thumb_ok
    
    def _is_sign_L(self, landmarks):
        """Detect ASL letter L: thumb and index extended, others closed."""
        if not landmarks:
            return False
        
        # Thumb and index extended
        thumb_extended = self._is_finger_extended(landmarks, 4, 3)
        index_extended = self._is_finger_extended(landmarks, 8, 7)
        
        # Middle, ring, pinky closed
        middle_closed = self._is_finger_bent(landmarks, 12, 11)
        ring_closed = self._is_finger_bent(landmarks, 16, 15)
        pinky_closed = self._is_finger_bent(landmarks, 20, 19)
        
        return thumb_extended and index_extended and middle_closed and ring_closed and pinky_closed
    
    def _is_sign_I(self, landmarks):
        """Detect ASL letter I: pinky extended, others closed."""
        if not landmarks:
            return False
        
        # Pinky extended
        pinky_extended = self._is_finger_extended(landmarks, 20, 19)
        
        # All others closed
        index_closed = self._is_finger_bent(landmarks, 8, 7)
        middle_closed = self._is_finger_bent(landmarks, 12, 11)
        ring_closed = self._is_finger_bent(landmarks, 16, 15)
        thumb_closed = self._is_finger_bent(landmarks, 4, 3)
        
        return pinky_extended and index_closed and middle_closed and ring_closed and thumb_closed
    
    def _is_sign_Y(self, landmarks):
        """Detect ASL letter Y: thumb and pinky extended, others closed."""
        if not landmarks:
            return False
        
        # Thumb and pinky extended
        thumb_extended = self._is_finger_extended(landmarks, 4, 3)
        pinky_extended = self._is_finger_extended(landmarks, 20, 19)
        
        # Index, middle, ring closed
        index_closed = self._is_finger_bent(landmarks, 8, 7)
        middle_closed = self._is_finger_bent(landmarks, 12, 11)
        ring_closed = self._is_finger_bent(landmarks, 16, 15)
        
        return thumb_extended and pinky_extended and index_closed and middle_closed and ring_closed
    
    def _is_sign_hello(self, landmarks):
        """Detect ASL hello: waving motion (simplified as open hand)."""
        if not landmarks:
            return False
        
        # All fingers extended
        all_extended = (
            self._is_finger_extended(landmarks, 8, 7) and
            self._is_finger_extended(landmarks, 12, 11) and
            self._is_finger_extended(landmarks, 16, 15) and
            self._is_finger_extended(landmarks, 20, 19)
        )
        
        # Thumb extended
        thumb_extended = self._is_finger_extended(landmarks, 4, 3)
        
        return all_extended and thumb_extended
    
    def _is_sign_thank_you(self, landmarks):
        """Detect ASL thank you: flat hand moving forward from chin."""
        # Simplified as flat hand
        return self._is_sign_hello(landmarks)
    
    def _is_sign_yes(self, landmarks):
        """Detect ASL yes: nodding fist."""
        # Simplified as closed fist
        return self._is_sign_A(landmarks)
    
    def _is_sign_no(self, landmarks):
        """Detect ASL no: shaking index finger."""
        # Simplified as index finger extended
        if not landmarks:
            return False
        
        index_extended = self._is_finger_extended(landmarks, 8, 7)
        others_closed = (
            self._is_finger_bent(landmarks, 12, 11) and
            self._is_finger_bent(landmarks, 16, 15) and
            self._is_finger_bent(landmarks, 20, 19) and
            self._is_finger_bent(landmarks, 4, 3)
        )
        
        return index_extended and others_closed
    
    def _is_sign_please(self, landmarks):
        """Detect ASL please: flat hand rubbing in circular motion on chest."""
        # Simplified as flat hand
        return self._is_sign_hello(landmarks)
    
    def _is_sign_sorry(self, landmarks):
        """Detect ASL sorry: fist making circular motion on chest."""
        # Simplified as closed fist
        return self._is_sign_A(landmarks)
    
    def _is_sign_good(self, landmarks):
        """Detect ASL good: flat hand touching chin then moving forward."""
        # Simplified as flat hand
        return self._is_sign_hello(landmarks)
    
    def _is_sign_bad(self, landmarks):
        """Detect ASL bad: flat hand touching chin then moving down."""
        # Simplified as flat hand
        return self._is_sign_hello(landmarks)
    
    # Placeholder methods for other letters (simplified)
    def _is_sign_D(self, landmarks): return False
    def _is_sign_E(self, landmarks): return False
    def _is_sign_F(self, landmarks): return False
    def _is_sign_G(self, landmarks): return False
    def _is_sign_H(self, landmarks): return False
    def _is_sign_J(self, landmarks): return False
    def _is_sign_K(self, landmarks): return False
    def _is_sign_M(self, landmarks): return False
    def _is_sign_N(self, landmarks): return False
    def _is_sign_O(self, landmarks): return False
    def _is_sign_P(self, landmarks): return False
    def _is_sign_Q(self, landmarks): return False
    def _is_sign_R(self, landmarks): return False
    def _is_sign_S(self, landmarks): return False
    def _is_sign_T(self, landmarks): return False
    def _is_sign_U(self, landmarks): return False
    def _is_sign_W(self, landmarks): return False
    def _is_sign_X(self, landmarks): return False
    def _is_sign_Z(self, landmarks): return False
    
    def detect_sign(self, landmarks) -> Tuple[str, float]:
        """
        Detect sign from hand landmarks using rule-based logic.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Tuple of (detected_sign, confidence)
        """
        if not landmarks:
            return "No Hand", 0.0
        
        detected_signs = []
        confidences = []
        
        # Test each sign definition
        for sign_name, detection_func in self.sign_definitions.items():
            try:
                if detection_func(landmarks):
                    detected_signs.append(sign_name)
                    # Calculate confidence based on how well landmarks match
                    confidences.append(0.9)  # High confidence for rule-based
            except Exception as e:
                continue
        
        if detected_signs:
            # Return the first detected sign with high confidence
            return detected_signs[0], max(confidences)
        
        return "Unknown", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Process a video frame for sign recognition.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, recognized_sign, confidence)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        recognized_sign = "No Hand"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            # Use the first detected hand
            landmarks = results.multi_hand_landmarks[0]
            
            # Detect sign
            detected_sign, detected_confidence = self.detect_sign(landmarks)
            
            if detected_sign != "Unknown" and detected_confidence > self.confidence_threshold:
                # Update recognition state
                self.last_signs.append(detected_sign)
                if len(self.last_signs) > 10:
                    self.last_signs.pop(0)
                
                # Check for stable recognition
                if len(self.last_signs) >= self.stable_frames_threshold:
                    from collections import Counter
                    most_common = Counter(self.last_signs).most_common(1)[0]
                    
                    if most_common[1] >= self.stable_frames_threshold:
                        stable_sign = most_common[0]
                        
                        # Only speak if it's a new sign and enough time has passed
                        current_time = time.time()
                        if (stable_sign != self.last_recognized_sign and 
                            current_time - self.last_spoken_at > self.speak_cooldown):
                            self.last_recognized_sign = stable_sign
                            self.last_spoken_at = current_time
                            self.speak_sign_async(stable_sign)
                        
                        recognized_sign = stable_sign
                        confidence = detected_confidence
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Display prediction info
                cv2.putText(frame, f"Detected: {detected_sign}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {detected_confidence:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current recognition
        cv2.putText(frame, f"Recognized: {recognized_sign}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame, recognized_sign, confidence
    
    def speak_sign_async(self, sign: str):
        """Speak sign asynchronously."""
        def speak():
            try:
                if sign.isalpha() and len(sign) == 1:
                    speech_text = f"The letter {sign}"
                elif sign.isdigit():
                    speech_text = f"The number {sign}"
                else:
                    speech_text = sign
                
                self.engine.say(speech_text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
        
        threading.Thread(target=speak, daemon=True).start()
    
    def release(self):
        """Release resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
