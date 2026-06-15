"""
Image Preprocessing Module
Handles all image preprocessing techniques for X-ray images.
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImagePreprocessor:
    """
    Handles image preprocessing for hairline fracture detection.
    """

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size (tuple): Target image size for model input
        """
        self.target_size = target_size
        self.preprocessing_steps = {}

    def load_image(self, image_path):
        """
        Load an image from file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            ndarray: Loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        return image

    def convert_to_grayscale(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.preprocessing_steps['grayscale'] = gray
        return gray

    def resize_image(self, image):
        """
        Resize image to target size.
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Resized image
        """
        resized = cv2.resize(image, self.target_size)
        self.preprocessing_steps['resized'] = resized
        return resized

    def apply_gaussian_blur(self, image, kernel_size=(5, 5), sigma=1.0):
        """
        Apply Gaussian Blur for noise reduction.
        
        Args:
            image (ndarray): Input image
            kernel_size (tuple): Size of the kernel
            sigma (float): Standard deviation
            
        Returns:
            ndarray: Blurred image
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        self.preprocessing_steps['gaussian_blur'] = blurred
        return blurred

    def apply_canny_edge_detection(self, image, threshold1=100, threshold2=200):
        """
        Apply Canny Edge Detection.
        
        Args:
            image (ndarray): Input image
            threshold1 (int): Lower threshold
            threshold2 (int): Upper threshold
            
        Returns:
            ndarray: Edge-detected image
        """
        edges = cv2.Canny(image, threshold1, threshold2)
        self.preprocessing_steps['canny_edges'] = edges
        return edges

    def apply_sobel_filtering(self, image):
        """
        Apply Sobel Filtering for edge enhancement.
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Sobel-filtered image
        """
        # Calculate Sobel derivatives
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
        # Calculate magnitude
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(255 * sobel / np.max(sobel))
        
        self.preprocessing_steps['sobel'] = sobel
        return sobel

    def normalize_image(self, image):
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Normalized image
        """
        normalized = image.astype(np.float32) / 255.0
        self.preprocessing_steps['normalized'] = normalized
        return normalized

    def preprocess_complete(self, image_path):
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            tuple: (processed_image, preprocessing_steps_dict)
        """
        self.preprocessing_steps = {}
        
        # Load image
        image = self.load_image(image_path)
        self.preprocessing_steps['original'] = image.copy()
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Resize
        resized = self.resize_image(gray)
        
        # Apply Gaussian Blur
        blurred = self.apply_gaussian_blur(resized)
        
        # Apply Canny Edge Detection
        edges = self.apply_canny_edge_detection(blurred)
        
        # Apply Sobel Filtering
        sobel = self.apply_sobel_filtering(blurred)
        
        # Combine preprocessing outputs (average the edges and sobel)
        combined = cv2.addWeighted(edges, 0.5, sobel, 0.5, 0)
        self.preprocessing_steps['combined'] = combined
        
        # Normalize
        normalized = self.normalize_image(combined)
        
        # Prepare final input (expand dimensions for model)
        final_input = np.expand_dims(normalized, axis=-1)
        final_input = np.expand_dims(final_input, axis=0)  # Add batch dimension
        
        return final_input, self.preprocessing_steps

    def preprocess_for_model(self, image_path):
        """
        Quick preprocessing for model prediction.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            ndarray: Preprocessed image ready for model
        """
        final_input, _ = self.preprocess_complete(image_path)
        return final_input
