#!/usr/bin/env python3
"""
Test script to check the sign recognition model status.
"""

import numpy as np
import tensorflow as tf
from sign_recognition import SignRecognizer

def test_model():
    """Test the sign recognition model."""
    print("🔍 Testing Sign Recognition Model...")
    
    # Initialize recognizer
    recognizer = SignRecognizer()
    
    print(f"\n📊 Model Info:")
    print(f"   - Model type: {type(recognizer.model)}")
    print(f"   - Input shape: {recognizer.model.input_shape}")
    print(f"   - Output shape: {recognizer.model.output_shape}")
    
    # Test with random input
    print(f"\n🧪 Testing with random input...")
    test_input = np.random.random((1, 47))  # 42 landmarks + 5 angles
    
    try:
        prediction = recognizer.model.predict(test_input, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        print(f"   - Raw prediction shape: {prediction.shape}")
        print(f"   - Predicted class: {predicted_class}")
        print(f"   - Confidence: {confidence:.4f}")
        print(f"   - Sign mapping: {recognizer.sign_mapping.get(predicted_class, 'Unknown')}")
        
        # Check if model is giving random predictions
        all_predictions = prediction[0]
        print(f"   - All class probabilities: {all_predictions[:10]}...")  # First 10 classes
        
        # Test multiple times to see if predictions are consistent
        print(f"\n🔄 Testing consistency...")
        predictions = []
        for i in range(5):
            test_input = np.random.random((1, 47))
            pred = recognizer.model.predict(test_input, verbose=0)
            pred_class = np.argmax(pred[0])
            predictions.append(pred_class)
        
        print(f"   - 5 random predictions: {predictions}")
        print(f"   - Unique predictions: {len(set(predictions))}")
        
        if len(set(predictions)) == 5:
            print("   ⚠️  Model is giving completely random predictions (untrained)")
        else:
            print("   ✅ Model shows some consistency")
            
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
    
    print(f"\n💡 Conclusion:")
    print(f"   - This model needs to be trained on sign language data")
    print(f"   - Without training, it will give random predictions")
    print(f"   - You need to either:")
    print(f"     1. Train the model with sign language data")
    print(f"     2. Use a pre-trained model")
    print(f"     3. Implement rule-based recognition as fallback")

if __name__ == "__main__":
    test_model()
