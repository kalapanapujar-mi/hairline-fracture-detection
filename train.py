"""
Model Training Module
Handles data loading, augmentation, training, and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from model import FractureDetectionModel
from preprocessing import ImagePreprocessor
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and prepare training data from folder structure."""

    def __init__(self, dataset_path='dataset', target_size=(224, 224), batch_size=32):
        """
        Initialize data loader.
        
        Args:
            dataset_path (str): Path to dataset folder
            target_size (tuple): Target image size
            batch_size (int): Batch size for training
        """
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def create_sample_dataset(self):
        """Create a sample dataset structure for demonstration."""
        import cv2
        
        dirs = [
            'dataset/train/fracture',
            'dataset/train/normal',
            'dataset/test/fracture',
            'dataset/test/normal'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print("Creating sample X-ray images...")
        
        # Fracture images (train)
        for i in range(15):
            img = np.random.randint(50, 150, (224, 224), dtype=np.uint8)
            for _ in range(3):
                x1, y1 = np.random.randint(0, 200), np.random.randint(0, 200)
                x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
                cv2.line(img, (x1, y1), (x2, y2), 255, 2)
            cv2.imwrite(f'dataset/train/fracture/fracture_{i}.png', img)
        
        # Normal images (train)
        for i in range(15):
            img = np.random.randint(100, 150, (224, 224), dtype=np.uint8)
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(f'dataset/train/normal/normal_{i}.png', img)
        
        # Fracture images (test)
        for i in range(5):
            img = np.random.randint(50, 150, (224, 224), dtype=np.uint8)
            for _ in range(3):
                x1, y1 = np.random.randint(0, 200), np.random.randint(0, 200)
                x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
                cv2.line(img, (x1, y1), (x2, y2), 255, 2)
            cv2.imwrite(f'dataset/test/fracture/fracture_{i}.png', img)
        
        # Normal images (test)
        for i in range(5):
            img = np.random.randint(100, 150, (224, 224), dtype=np.uint8)
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(f'dataset/test/normal/normal_{i}.png', img)
        
        print("Sample dataset created successfully!")

    def load_data(self, augment=True):
        """
        Load data using ImageDataGenerator.
        
        Args:
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        if not os.path.exists(os.path.join(self.dataset_path, 'train')):
            print("Dataset not found. Creating sample dataset...")
            self.create_sample_dataset()
        
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_data = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes={'fracture': 0, 'normal': 1}
        )
        
        val_path = os.path.join(self.dataset_path, 'val')
        if os.path.exists(val_path):
            self.val_data = val_datagen.flow_from_directory(
                val_path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes={'fracture': 0, 'normal': 1}
            )
        else:
            self.val_data = None
        
        self.test_data = test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            classes={'fracture': 0, 'normal': 1}
        )
        
        return self.train_data, self.val_data, self.test_data


class ModelTrainer:
    """Handle model training and evaluation."""

    def __init__(self, model, data_loader):
        """Initialize trainer."""
        self.model = model
        self.data_loader = data_loader
        self.history = None
        self.evaluation_metrics = {}

    def train(self, epochs=50, verbose=1):
        """Train the model."""
        train_data, val_data, _ = self.data_loader.load_data(augment=True)
        
        callbacks = [
            EarlyStopping(
                monitor='loss' if val_data is None else 'val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss' if val_data is None else 'val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        steps_per_epoch = max(1, len(train_data))
        
        self.history = self.model.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            steps_per_epoch=steps_per_epoch
        )
        
        return self.history

    def evaluate(self):
        """Evaluate model on test set."""
        _, _, test_data = self.data_loader.load_data(augment=False)
        
        y_pred = []
        y_true = []
        
        for images, labels in test_data:
            predictions = self.model.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels, axis=1))
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'confusion_matrix': np.array([[0, 0], [0, 0]]),
                'classification_report': 'No test data'
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
        recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, target_names=['Fracture', 'Normal'], zero_division=0)
        }
        
        return self.evaluation_metrics
