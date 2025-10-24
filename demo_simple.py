#!/usr/bin/env python3
"""
Simple demo script for the OpenCV-based Sign Language Recognition system.
Compatible with Python 3.13 and ARM64 systems.
"""

import cv2
import numpy as np
import time
from hand_tracking_opencv import HandTrackerOpenCV
from sign_recognition_simple import SimpleSignRecognizer

def demo_hand_tracking():
    """Demonstrate OpenCV hand tracking functionality."""
    print("=== OpenCV Hand Tracking Demo ===")
    print("This demo shows hand detection using OpenCV methods.")
    print("Press 'q' to quit, 's' to save a frame")
    
    tracker = HandTrackerOpenCV()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
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
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Hands: {len(hand_features)}", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_features:
                cv2.putText(processed_frame, f"Features: {len(hand_features[0])}", (10, 330), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('OpenCV Hand Tracking Demo', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f"opencv_hand_tracking_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("OpenCV hand tracking demo completed")

def demo_sign_recognition():
    """Demonstrate simplified sign recognition functionality."""
    print("\n=== Simple Sign Recognition Demo ===")
    print("This demo shows sign recognition with audio output.")
    print("Press 'q' to quit, 's' to speak current sign")
    
    # Initialize recognizer
    recognizer = SimpleSignRecognizer()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
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
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Auto-speak new signs (with cooldown)
            if (sign != last_spoken_sign and 
                sign != "No Sign" and 
                confidence > 0.5 and
                time.time() - start_time > 5):  # Wait 5 seconds before auto-speaking
                recognizer.speak_sign_async(sign)
                last_spoken_sign = sign
            
            # Display frame
            cv2.imshow('Simple Sign Recognition Demo', processed_frame)
            
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
        print("Simple sign recognition demo completed")

def demo_feature_extraction():
    """Demonstrate feature extraction from OpenCV hand detection."""
    print("\n=== Feature Extraction Demo ===")
    print("This demo shows the numerical features extracted from hand detection.")
    print("Press 'q' to quit, 'f' to show features")
    
    tracker = HandTrackerOpenCV()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    show_features = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features = tracker.detect_hands(frame)
            
            # Display basic info
            cv2.putText(processed_frame, f"Hands: {len(hand_features)}", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_features:
                features = hand_features[0]
                cv2.putText(processed_frame, f"Features: {len(features)}", (10, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show feature values if requested
                if show_features:
                    # Display first few feature values
                    for i in range(min(10, len(features))):
                        y_pos = 330 + i * 20
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

def show_help():
    """Show help information."""
    print("\nOpenCV Sign Language Recognition System - Demo")
    print("=" * 50)
    print("This demo showcases the simplified OpenCV-based system.")
    print("Make sure you have a webcam connected and good lighting.")
    print()
    print("Available demos:")
    print("1. Hand Tracking Demo - Basic hand detection")
    print("2. Sign Recognition Demo - Full recognition with audio")
    print("3. Feature Extraction Demo - View extracted features")
    print("4. Help - Show this help message")
    print("5. Exit")
    print()
    print("For best results:")
    print("- Ensure good lighting")
    print("- Keep hands clearly visible in camera")
    print("- Use a good quality webcam")
    print("- Make clear hand gestures")
    print()
    print("Note: This is a simplified version using OpenCV instead of MediaPipe.")
    print("Accuracy may be lower but it's compatible with Python 3.13 + ARM64.")

def main():
    """Main demo function."""
    print("OpenCV Sign Language Recognition System - Demo")
    print("=" * 50)
    print("This demo showcases the simplified OpenCV-based system.")
    print("Compatible with Python 3.13 and ARM64 systems.")
    print()
    
    while True:
        print("\nAvailable demos:")
        print("1. Hand Tracking Demo")
        print("2. Sign Recognition Demo")
        print("3. Feature Extraction Demo")
        print("4. Help")
        print("5. Exit")
        
        try:
            choice = input("\nSelect a demo (1-5): ").strip()
            
            if choice == '1':
                demo_hand_tracking()
            elif choice == '2':
                demo_sign_recognition()
            elif choice == '3':
                demo_feature_extraction()
            elif choice == '4':
                show_help()
            elif choice == '5':
                print("Exiting demo...")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
    
    print("Demo completed. Thank you for trying the OpenCV-based system!")

if __name__ == '__main__':
    main()
