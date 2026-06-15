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


class DataLoader:
    """
    Load and prepare training data from folder structure.
    """

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
        """
        Create a sample dataset structure for demonstration.
        This should be replaced with actual X-ray images.
        """
        import cv2
        from PIL import Image
        
        # Create directory structure
        dirs = [
            'dataset/train/fracture',
            'dataset/train/normal',
            'dataset/test/fracture',
            'dataset/test/normal'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create sample images
        print("Creating sample X-ray images...")
        
        # Fracture images (train)
        for i in range(10):
            img = np.random.randint(50, 150, (224, 224), dtype=np.uint8)
            # Add fracture-like patterns
            for _ in range(3):
                x1, y1 = np.random.randint(0, 200), np.random.randint(0, 200)
                x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
                cv2.line(img, (x1, y1), (x2, y2), 255, 2)
            cv2.imwrite(f'dataset/train/fracture/fracture_{i}.png', img)
        
        # Normal images (train)
        for i in range(10):
            img = np.random.randint(100, 150, (224, 224), dtype=np.uint8)
            # Add some Gaussian noise for texture
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
        
        # Load training data
        self.train_data = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        # Load validation data (if exists)
        val_path = os.path.join(self.dataset_path, 'val')
        if os.path.exists(val_path):
            self.val_data = val_datagen.flow_from_directory(
                val_path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='categorical'
            )
        
        # Load test data
        self.test_data = test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return self.train_data, self.val_data, self.test_data


class ModelTrainer:
    """
    Handle model training and evaluation.
    """

    def __init__(self, model, data_loader):
        """
        Initialize trainer.
        
        Args:
            model (FractureDetectionModel): Model instance
            data_loader (DataLoader): Data loader instance
        """
        self.model = model
        self.data_loader = data_loader
        self.history = None
        self.evaluation_metrics = {}

    def train(self, epochs=50, verbose=1):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            verbose (int): Verbosity level
            
        Returns:
            History: Training history
        """
        train_data, val_data, _ = self.data_loader.load_data(augment=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history

    def evaluate(self):
        """
        Evaluate model on test set.
        
        Returns:
            dict: Evaluation metrics
        """
        _, _, test_data = self.data_loader.load_data(augment=False)
        
        # Get predictions
        y_pred = []
        y_true = []
        
        for images, labels in test_data:
            predictions = self.model.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels, axis=1))
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, target_names=['Fracture', 'Normal'])
        }
        
        return self.evaluation_metrics

    def plot_training_history(self):
        """
        Plot training and validation metrics.
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plot confusion matrix.
        """
        if 'confusion_matrix' not in self.evaluation_metrics:
            print("No evaluation metrics available.")
            return
        
        cm = self.evaluation_metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Fracture', 'Normal'])
        plt.yticks(tick_marks, ['Fracture', 'Normal'])
        
        # Add values to cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix plot saved as 'confusion_matrix.png'")
        plt.show()

    def print_evaluation_metrics(self):
        """
        Print evaluation metrics.
        """
        if not self.evaluation_metrics:
            print("No evaluation metrics available.")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {self.evaluation_metrics['accuracy']:.4f}")
        print(f"Precision: {self.evaluation_metrics['precision']:.4f}")
        print(f"Recall:    {self.evaluation_metrics['recall']:.4f}")
        print(f"F1-Score:  {self.evaluation_metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(self.evaluation_metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(self.evaluation_metrics['classification_report'])
        print("="*50)
