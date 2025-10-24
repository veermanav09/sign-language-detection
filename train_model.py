import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from hand_tracking import HandTracker
import cv2
import argparse
from typing import Tuple, List, Dict

class SignLanguageTrainer:
    """
    Trainer for sign language recognition model.
    Handles data collection, preprocessing, training, and evaluation.
    """
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.hand_tracker = HandTracker()
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        self.learning_rate = 0.001
        
        # Data storage
        self.features = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        # Model architecture parameters
        self.input_shape = (47,)  # 42 landmarks + 5 angles
        self.num_classes = 0
        
    def collect_data_from_video(self, sign_name: str, video_path: str, max_frames: int = 100):
        """
        Collect hand features from a video file for a specific sign.
        
        Args:
            sign_name: Name of the sign being collected
            video_path: Path to the video file
            max_frames: Maximum number of frames to process
        """
        print(f"Collecting data for sign: {sign_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        collected_features = []
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features_list = self.hand_tracker.detect_hands(frame)
            
            if hand_features_list:
                # Use the first detected hand
                features = hand_features_list[0]
                
                # Ensure features are the right shape
                if len(features) == 47:
                    collected_features.append(features)
                    frame_count += 1
                    
                    # Display progress
                    if frame_count % 10 == 0:
                        print(f"  Collected {frame_count} frames")
            
            # Show frame for debugging
            cv2.imshow(f'Data Collection - {sign_name}', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected features
        if collected_features:
            features_array = np.array(collected_features)
            features_file = os.path.join(self.data_dir, f"{sign_name}_features.npy")
            np.save(features_file, features_array)
            
            print(f"  Saved {len(collected_features)} feature samples to {features_file}")
            
            # Add to training data
            self.features.extend(collected_features)
            self.labels.extend([sign_name] * len(collected_features))
        else:
            print(f"  No valid features collected for {sign_name}")
    
    def collect_data_from_camera(self, sign_name: str, duration: int = 30, fps: int = 10):
        """
        Collect hand features from live camera for a specific sign.
        
        Args:
            sign_name: Name of the sign being collected
            duration: Duration in seconds to collect data
            fps: Frames per second to collect
        """
        print(f"Collecting data for sign: {sign_name}")
        print(f"Please perform the sign '{sign_name}' for {duration} seconds")
        print("Press 'q' to stop early")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        frame_count = 0
        collected_features = []
        start_time = cv2.getTickCount()
        
        while frame_count < duration * fps:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_features_list = self.hand_tracker.detect_hands(frame)
            
            # Add countdown and instructions
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            remaining_time = duration - elapsed_time
            
            cv2.putText(processed_frame, f"Sign: {sign_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Time: {remaining_time:.1f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Frames: {frame_count}/{duration * fps}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if hand_features_list:
                # Use the first detected hand
                features = hand_features_list[0]
                
                # Ensure features are the right shape
                if len(features) == 47:
                    collected_features.append(features)
                    frame_count += 1
                    
                    # Add green circle for successful collection
                    cv2.circle(processed_frame, (600, 50), 20, (0, 255, 0), -1)
            
            # Show frame
            cv2.imshow(f'Data Collection - {sign_name}', processed_frame)
            
            # Check for early exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate
            cv2.waitKey(int(1000 / fps))
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected features
        if collected_features:
            features_array = np.array(collected_features)
            features_file = os.path.join(self.data_dir, f"{sign_name}_features.npy")
            np.save(features_file, features_array)
            
            print(f"  Collected {len(collected_features)} feature samples")
            print(f"  Saved to {features_file}")
            
            # Add to training data
            self.features.extend(collected_features)
            self.labels.extend([sign_name] * len(collected_features))
        else:
            print(f"  No valid features collected for {sign_name}")
    
    def load_existing_data(self):
        """Load existing feature files from the data directory."""
        print("Loading existing training data...")
        
        feature_files = [f for f in os.listdir(self.data_dir) if f.endswith('_features.npy')]
        
        if not feature_files:
            print("No existing feature files found")
            return
        
        for feature_file in feature_files:
            sign_name = feature_file.replace('_features.npy', '')
            features_path = os.path.join(self.data_dir, feature_file)
            
            try:
                features = np.load(features_path)
                self.features.extend(features)
                self.labels.extend([sign_name] * len(features))
                print(f"  Loaded {len(features)} samples for sign '{sign_name}'")
            except Exception as e:
                print(f"  Error loading {feature_file}: {e}")
        
        print(f"Total samples loaded: {len(self.features)}")
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the collected data for training.
        
        Returns:
            Tuple of (features, labels) ready for training
        """
        if not self.features or not self.labels:
            raise ValueError("No data loaded. Please collect or load data first.")
        
        print("Preprocessing data...")
        
        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y_encoded.shape}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        
        # Normalize features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        return X, y_encoded
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the neural network model for sign recognition.
        
        Returns:
            Compiled Keras model
        """
        print("Creating model...")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape),
            
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
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"  Model created with {model.count_params():,} parameters")
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Trained model
        """
        print("Training model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Create model
        model = self.create_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        history_file = os.path.join(self.model_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        print(f"  Training completed. Best model saved to {os.path.join(self.model_dir, 'best_model.h5')}")
        
        return model, history
    
    def evaluate_model(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            X: Test features
            y: Test labels
        """
        print("Evaluating model...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test loss: {test_loss:.4f}")
        
        # Predictions for detailed analysis
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_file = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Confusion matrix saved to {cm_file}")
        
        # Classification report
        report = classification_report(y_test, y_pred_classes, 
                                    target_names=self.label_encoder.classes_,
                                    output_dict=True)
        
        report_file = os.path.join(self.model_dir, 'classification_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  Classification report saved to {report_file}")
        
        return test_accuracy, test_loss
    
    def save_model_and_metadata(self, model: tf.keras.Model):
        """Save the trained model and metadata."""
        print("Saving model and metadata...")
        
        # Save model
        model_file = os.path.join(self.model_dir, 'sign_recognition_model.h5')
        model.save(model_file)
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist(),
            'feature_names': [f'landmark_{i//2}_{"x" if i%2==0 else "y"}' for i in range(42)] + 
                           [f'angle_{i}' for i in range(5)],
            'training_samples': len(self.features),
            'model_architecture': 'Sequential with Dense layers and Dropout'
        }
        
        metadata_file = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoder
        encoder_file = os.path.join(self.model_dir, 'label_encoder.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"  Model saved to {model_file}")
        print(f"  Metadata saved to {metadata_file}")
        print(f"  Label encoder saved to {encoder_file}")
    
    def interactive_data_collection(self):
        """Interactive data collection from camera."""
        print("Interactive Data Collection Mode")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Collect data for a new sign")
            print("2. View collected data summary")
            print("3. Start training")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                sign_name = input("Enter sign name: ").strip()
                if sign_name:
                    duration = int(input("Enter collection duration in seconds (default 30): ") or "30")
                    self.collect_data_from_camera(sign_name, duration)
            
            elif choice == '2':
                self.print_data_summary()
            
            elif choice == '3':
                if len(self.features) > 0:
                    self.start_training()
                else:
                    print("No data collected yet. Please collect some data first.")
            
            elif choice == '4':
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def print_data_summary(self):
        """Print summary of collected data."""
        if not self.features:
            print("No data collected yet.")
            return
        
        print("\nData Summary:")
        print("=" * 30)
        
        from collections import Counter
        label_counts = Counter(self.labels)
        
        for sign, count in label_counts.items():
            print(f"  {sign}: {count} samples")
        
        print(f"\nTotal samples: {len(self.features)}")
        print(f"Feature dimensions: {len(self.features[0])}")
    
    def start_training(self):
        """Start the training process."""
        try:
            # Preprocess data
            X, y = self.preprocess_data()
            
            # Train model
            model, history = self.train_model(X, y)
            
            # Evaluate model
            accuracy, loss = self.evaluate_model(model, X, y)
            
            # Save everything
            self.save_model_and_metadata(model)
            
            print(f"\nTraining completed successfully!")
            print(f"Final test accuracy: {accuracy:.4f}")
            print(f"Model and metadata saved to {self.model_dir}/")
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Train sign language recognition model')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                       help='Training mode: interactive or batch')
    parser.add_argument('--data-dir', default='data', help='Directory for training data')
    parser.add_argument('--model-dir', default='models', help='Directory for trained models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SignLanguageTrainer(args.data_dir, args.model_dir)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    
    if args.mode == 'interactive':
        trainer.interactive_data_collection()
    else:
        # Load existing data and train
        trainer.load_existing_data()
        if len(trainer.features) > 0:
            trainer.start_training()
        else:
            print("No training data found. Please collect data first.")

if __name__ == '__main__':
    main()
