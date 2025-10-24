#!/usr/bin/env python3
"""
Test script for the rule-based sign recognizer.
"""

import cv2
import numpy as np
from sign_recognition_rule_based import RuleBasedSignRecognizer

def test_rule_based_recognizer():
    """Test the rule-based sign recognizer with camera."""
    print("🔍 Testing Rule-Based Sign Recognizer...")
    print("Press 'q' to quit")
    
    # Initialize recognizer
    recognizer = RuleBasedSignRecognizer()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("📝 Make hand signs in front of the camera:")
    print("   - A: Closed fist with thumb extended")
    print("   - B: All fingers extended, thumb tucked")
    print("   - C: Curved hand like holding a cup")
    print("   - V: Index and middle extended, others closed")
    print("   - L: Thumb and index extended, others closed")
    print("   - I: Pinky extended, others closed")
    print("   - Y: Thumb and pinky extended, others closed")
    print("   - Hello: All fingers extended")
    print("   - No: Index finger extended, others closed")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame, sign, confidence = recognizer.process_frame(frame)
            
            # Display frame
            cv2.imshow('Rule-Based Sign Recognition Test', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.release()
        print("✅ Test completed")

if __name__ == "__main__":
    test_rule_based_recognizer()
