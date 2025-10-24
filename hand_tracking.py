import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandTracker:
    """
    Hand tracking and feature extraction using MediaPipe.
    Extracts hand landmarks and converts them to features for sign recognition.
    """
    
    def __init__(self):
        """Initialize MediaPipe hand tracking."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection with optimized parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support for both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Hand landmark indices for key points
        self.landmark_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],
            'wrist': [0]
        }
    
    def extract_hand_features(self, landmarks) -> np.ndarray:
        """
        Extract numerical features from hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            numpy array of features (42 values: 21 landmarks × 2 coordinates)
        """
        if landmarks is None:
            return np.zeros(42)
        
        features = []
        for landmark in landmarks.landmark:
            # Normalize coordinates to [0, 1] range
            features.extend([landmark.x, landmark.y])
        
        return np.array(features)
    
    def calculate_hand_angles(self, landmarks) -> List[float]:
        """
        Calculate angles between finger joints for additional features.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            List of angles in degrees
        """
        if landmarks is None:
            return []
        
        angles = []
        
        # Calculate angles for each finger
        for finger_name, indices in self.landmark_indices.items():
            if finger_name == 'wrist':
                continue
                
            if len(indices) >= 3:
                # Get three consecutive points for angle calculation
                p1 = np.array([landmarks.landmark[indices[0]].x, landmarks.landmark[indices[0]].y])
                p2 = np.array([landmarks.landmark[indices[1]].x, landmarks.landmark[indices[1]].y])
                p3 = np.array([landmarks.landmark[indices[2]].x, landmarks.landmark[indices[2]].y])
                
                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                angles.append(angle)
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points.
        
        Args:
            p1, p2, p3: 2D points as numpy arrays
            
        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect hands in the image and extract features.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (processed_image, hand_features_list)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        processed_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        hand_features = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    processed_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract features
                features = self.extract_hand_features(hand_landmarks)
                angles = self.calculate_hand_angles(hand_landmarks)
                
                # Combine features
                combined_features = np.concatenate([features, angles])
                hand_features.append(combined_features)
        
        return processed_image, hand_features
    
    def get_hand_bbox(self, landmarks) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box coordinates for the hand.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Tuple of (x, y, width, height) or None if no landmarks
        """
        if landmarks is None:
            return None
        
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to pixel coordinates (assuming image is 640x480)
        x_min = int(x_min * 640)
        x_max = int(x_max * 640)
        y_min = int(y_min * 480)
        y_max = int(y_max * 480)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def is_hand_visible(self, landmarks) -> bool:
        """
        Check if hand is clearly visible in the frame.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            True if hand is visible, False otherwise
        """
        if landmarks is None:
            return False
        
        # Check if all key landmarks are detected
        key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
        for idx in key_landmarks:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                # Check if landmark is within reasonable bounds
                if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                    return False
            else:
                return False
        
        return True
    
    def release(self):
        """Release resources."""
        self.hands.close()

# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Hand tracking started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features = tracker.detect_hands(frame)
            
            # Display hand count
            hand_count = len(hand_features)
            cv2.putText(processed_frame, f"Hands: {hand_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display features info
            if hand_features:
                feature_count = len(hand_features[0])
                cv2.putText(processed_frame, f"Features: {feature_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Hand Tracking', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
