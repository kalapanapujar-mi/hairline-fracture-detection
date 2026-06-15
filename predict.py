"""
Prediction Module
Handles image prediction and result generation.
"""

import numpy as np
from preprocessing import ImagePreprocessor
from model import FractureDetectionModel


class FracturePredictor:
    """
    Predict hairline fractures in X-ray images.
    """

    def __init__(self, model_path='fracture_model.h5'):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to the trained model
        """
        self.preprocessor = ImagePreprocessor()
        self.model = FractureDetectionModel(model_path=model_path)
        self.last_preprocessing_steps = None
        self.last_prediction = None

    def load_model(self):
        """
        Load the trained model.
        
        Returns:
            bool: True if successful
        """
        try:
            self.model.load_model()
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False

    def predict_image(self, image_path):
        """
        Predict on a single image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Prediction results
        """
        try:
            # Preprocess image
            processed_image, preprocessing_steps = self.preprocessor.preprocess_complete(image_path)
            self.last_preprocessing_steps = preprocessing_steps
            
            # Make prediction
            prediction_text, confidence, probabilities = self.model.predict(processed_image)
            
            self.last_prediction = {
                'prediction': prediction_text,
                'confidence': confidence,
                'confidence_percentage': confidence * 100,
                'fracture_probability': probabilities[0],
                'normal_probability': probabilities[1],
                'preprocessing_steps': preprocessing_steps
            }
            
            return self.last_prediction
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def get_prediction_summary(self):
        """
        Get a text summary of the last prediction.
        
        Returns:
            str: Prediction summary
        """
        if self.last_prediction is None:
            return "No prediction made yet."
        
        summary = f"""
        ===== PREDICTION RESULT =====
        Result: {self.last_prediction['prediction']}
        Confidence: {self.last_prediction['confidence_percentage']:.2f}%
        
        Fracture Probability: {self.last_prediction['fracture_probability']:.4f}
        Normal Probability: {self.last_prediction['normal_probability']:.4f}
        ============================
        """
        return summary

    def get_preprocessing_steps(self):
        """
        Get the preprocessing steps from last prediction.
        
        Returns:
            dict: Dictionary of preprocessing steps
        """
        return self.last_preprocessing_steps
