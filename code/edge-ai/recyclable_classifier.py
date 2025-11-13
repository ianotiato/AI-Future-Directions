Edge AI Recyclable Items Classifier
Using TensorFlow Lite for lightweight image classification
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import os

class EdgeAIClassifier:
    def __init__(self):
        self.model = None
        self.tflite_model = None
        self.class_names = ['plastic', 'paper', 'glass', 'metal', 'cardboard']
        
    def create_model(self):
        """Create a lightweight CNN model for edge deployment"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(5, activation='softmax')  # 5 recyclable categories
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic image data for demonstration"""
        # In real scenario, use datasets like Kaggle Waste Classification
        x_train = np.random.random((num_samples, 128, 128, 3)).astype(np.float32)
        y_train = np.random.randint(0, 5, (num_samples,))
        
        x_test = np.random.random((200, 128, 128, 3)).astype(np.float32)
        y_test = np.random.randint(0, 5, (200,))
        
        return (x_train, y_train), (x_test, y_test)
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10):
        """Train the model"""
        history = self.model.fit(x_train, y_train,
                                epochs=epochs,
                                validation_data=(x_test, y_test))
        return history
    
    def convert_to_tflite(self):
        """Convert model to TensorFlow Lite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.tflite_model = converter.convert()
        
        # Save the TFLite model
        with open('recyclable_classifier.tflite', 'wb') as f:
            f.write(self.tflite_model)
        
        return self.tflite_model
    
    def test_tflite_model(self, test_data):
        """Test the TFLite model"""
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test with sample data
        input_data = test_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    
    def evaluate_edge_ai_benefits(self):
        """Analyze benefits of Edge AI implementation"""
        benefits = {
            "Latency": "~10-50ms vs 200-1000ms cloud processing",
            "Privacy": "Data processed locally, no external transmission",
            "Bandwidth": "Minimal cloud dependency, reduced data transfer",
            "Reliability": "Works offline, robust to network issues",
            "Power Consumption": "Optimized for edge devices"
        }
        return benefits

def main():
    # Initialize classifier
    classifier = EdgeAIClassifier()
    
    # Create and display model architecture
    model = classifier.create_model()
    print("Model Architecture:")
    model.summary()
    
    # Generate synthetic data
    (x_train, y_train), (x_test, y_test) = classifier.generate_synthetic_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Train model
    print("\nTraining model...")
    history = classifier.train_model(x_train, y_train, x_test, y_test, epochs=5)
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    tflite_model = classifier.convert_to_tflite()
    print(f"TFLite model size: {len(tflite_model)} bytes")
    
    # Test TFLite model
    sample_data = x_test[:1]
    predictions = classifier.test_tflite_model(sample_data)
    print(f"\nSample prediction: {predictions}")
    print(f"Predicted class: {classifier.class_names[np.argmax(predictions)]}")
    
    # Display benefits
    benefits = classifier.evaluate_edge_ai_benefits()
    print("\nEdge AI Benefits Analysis:")
    for benefit, description in benefits.items():
        print(f"- {benefit}: {description}")

if __name__ == "__main__":
    main()
