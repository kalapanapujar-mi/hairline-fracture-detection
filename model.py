"""
Convolutional Neural Network Model for Hairline Fracture Detection
Defines and manages the CNN architecture.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os


class FractureDetectionModel:
    """
    CNN model for detecting hairline fractures in X-ray images.
    """

    def __init__(self, model_path='fracture_model.h5'):
        """
        Initialize the model.
        
        Args:
            model_path (str): Path to save/load the model
        """
        self.model_path = model_path
        self.model = None
        self.history = None

    def build_model(self, input_shape=(224, 224, 1)):
        """
        Build the CNN architecture.
        
        Args:
            input_shape (tuple): Shape of input images
            
        Returns:
            Model: Compiled Keras model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(2, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        self.model = model
        return model

    def get_model_summary(self):
        """
        Get model summary.
        
        Returns:
            str: Model architecture summary
        """
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

    def load_model(self):
        """
        Load pre-trained model from disk.
        
        Returns:
            Model: Loaded Keras model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return self.model

    def save_model(self):
        """
        Save model to disk.
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, image_data):
        """
        Make prediction on input image.
        
        Args:
            image_data (ndarray): Preprocessed image data
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or build_model() first.")
        
        # Ensure input has correct shape
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_data, verbose=0)
        
        # Get class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Class labels: 0 = Fracture, 1 = Normal
        class_labels = ['Fracture Detected', 'No Fracture Detected']
        prediction_text = class_labels[predicted_class]
        
        return prediction_text, confidence, predictions[0]

    def get_model_info(self):
        """
        Get detailed model information.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "Model not built yet"}
        
        return {
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            "layers": len(self.model.layers),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape
        }
