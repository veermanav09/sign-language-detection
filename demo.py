#!/usr/bin/env python3
"""
Simple demo script for the Sign Language Recognition system.
This script provides a basic demonstration of the core functionality.
"""

import cv2
import numpy as np
import time
from hand_tracking import HandTracker
from sign_recognition import SignRecognizer

def demo_hand_tracking():
    """Demonstrate hand tracking functionality."""
    print("=== Hand Tracking Demo ===")
    print("This demo shows hand landmark detection using MediaPipe.")
    print("Press 'q' to quit, 's' to save a frame")
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features = tracker.detect_hands(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display info
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Hands: {len(hand_features)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_features:
                cv2.putText(processed_frame, f"Features: {len(hand_features[0])}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show first hand's bounding box
                bbox = tracker.get_hand_bbox(hand_features[0])
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Hand Tracking Demo', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f"hand_tracking_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("Hand tracking demo completed")

def demo_sign_recognition():
    """Demonstrate sign recognition functionality."""
    print("\n=== Sign Recognition Demo ===")
    print("This demo shows sign recognition with audio output.")
    print("Press 'q' to quit, 's' to speak current sign")
    
    # Initialize recognizer (without pre-trained model)
    recognizer = SignRecognizer()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    last_spoken_sign = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, sign, confidence = recognizer.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display info
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Sign: {sign}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(processed_frame, f"Confidence: {confidence:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Auto-speak new signs (with cooldown)
            if (sign != last_spoken_sign and 
                sign != "No Sign" and 
                confidence > 0.7 and
                time.time() - start_time > 5):  # Wait 5 seconds before auto-speaking
                recognizer.speak_sign_async(sign)
                last_spoken_sign = sign
            
            # Display frame
            cv2.imshow('Sign Recognition Demo', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manually speak current sign
                if sign != "No Sign":
                    recognizer.speak_sign_async(sign)
                    print(f"Speaking: {sign}")
                else:
                    print("No sign detected to speak")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.hand_tracker.release()
        print("Sign recognition demo completed")

def demo_feature_extraction():
    """Demonstrate feature extraction from hand landmarks."""
    print("\n=== Feature Extraction Demo ===")
    print("This demo shows the numerical features extracted from hand landmarks.")
    print("Press 'q' to quit, 'f' to show features")
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    show_features = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features = tracker.detect_hands(frame)
            
            # Display basic info
            cv2.putText(processed_frame, f"Hands: {len(hand_features)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_features:
                features = hand_features[0]
                cv2.putText(processed_frame, f"Features: {len(features)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show feature values if requested
                if show_features:
                    # Display first few feature values
                    for i in range(min(10, len(features))):
                        y_pos = 90 + i * 20
                        cv2.putText(processed_frame, f"F{i}: {features[i]:.3f}", 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Feature Extraction Demo', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_features = not show_features
                print(f"Feature display: {'ON' if show_features else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("Feature extraction demo completed")

def main():
    """Main demo function."""
    print("Sign Language Recognition System - Demo")
    print("=" * 50)
    print("This demo showcases the core functionality of the system.")
    print("Make sure you have a webcam connected and good lighting.")
    print()
    
    while True:
        print("\nAvailable demos:")
        print("1. Hand Tracking Demo")
        print("2. Sign Recognition Demo")
        print("3. Feature Extraction Demo")
        print("4. Exit")
        
        choice = input("\nSelect a demo (1-4): ").strip()
        
        if choice == '1':
            demo_hand_tracking()
        elif choice == '2':
            demo_sign_recognition()
        elif choice == '3':
            demo_feature_extraction()
        elif choice == '4':
            print("Exiting demo...")
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("Demo completed. Thank you for trying the system!")

if __name__ == '__main__':
    main()
