"""
Main GUI Application for Hairline Fracture Detection
Desktop application using Tkinter for X-ray image analysis.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from predict import FracturePredictor
from model import FractureDetectionModel
from train import DataLoader, ModelTrainer
import os
import threading


class FractureDetectionApp:
    """
    Main GUI application for fracture detection.
    """

    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root (tk.Tk): Root window
        """
        self.root = root
        self.root.title("Hairline Fracture Detection System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Initialize predictor
        self.predictor = FracturePredictor()
        self.model_loaded = False
        
        # Setup GUI
        self.setup_ui()
        
        # Try to load model
        self.load_model()

    def setup_ui(self):
        """
        Setup the user interface.
        """
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            main_container,
            text="Hairline Fracture Detection System",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.upload_btn = ttk.Button(
            button_frame,
            text="Upload X-ray Image",
            command=self.upload_image
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(
            button_frame,
            text="Train Model",
            command=self.open_training_window
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Model status
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(status_frame, text="Model Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(
            status_frame,
            text="Not Loaded",
            foreground="red",
            font=("Arial", 10, "bold")
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Original image
        left_panel = ttk.LabelFrame(content_frame, text="Original Image", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(left_panel, bg="gray", height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Middle panel - Preprocessing steps
        middle_panel = ttk.LabelFrame(content_frame, text="Preprocessing Steps", padding=10)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Preprocessing steps notebook
        self.preprocessing_notebook = ttk.Notebook(middle_panel)
        self.preprocessing_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(content_frame, text="Prediction Results", padding=10)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Result display
        self.result_text = tk.Text(right_panel, height=30, width=40, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(
            main_container,
            text="Ready",
            relief=tk.SUNKEN
        )
        self.status_bar.pack(fill=tk.X, pady=5)
        
        self.current_image_path = None

    def load_model(self):
        """
        Load the trained model.
        """
        try:
            if self.predictor.load_model():
                self.model_loaded = True
                self.status_label.config(text="Loaded", foreground="green")
                self.predict_btn.config(state=tk.NORMAL)
                self.status_bar.config(text="Model loaded successfully")
            else:
                self.status_label.config(text="Not Found", foreground="red")
                self.status_bar.config(text="Model file not found. Train a new model or place fracture_model.h5 in the current directory.")
        except Exception as e:
            self.status_label.config(text="Error", foreground="red")
            self.status_bar.config(text=f"Error loading model: {e}")

    def upload_image(self):
        """
        Upload an image file.
        """
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_original_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")

    def display_original_image(self, image_path):
        """
        Display the original image.
        
        Args:
            image_path (str): Path to the image
        """
        image = Image.open(image_path)
        
        # Resize for display
        display_size = (400, 400)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.original_canvas.create_image(
            self.original_canvas.winfo_width()//2,
            self.original_canvas.winfo_height()//2,
            image=photo
        )
        self.original_canvas.image = photo

    def predict(self):
        """
        Make a prediction on the loaded image.
        """
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        if not self.model_loaded:
            messagebox.showerror("Error", "Model is not loaded.")
            return
        
        self.status_bar.config(text="Predicting...")
        self.root.update()
        
        try:
            # Make prediction
            result = self.predictor.predict_image(self.current_image_path)
            
            if result:
                # Display results
                self.display_results(result)
                self.display_preprocessing_steps(result['preprocessing_steps'])
                self.status_bar.config(text="Prediction complete")
            else:
                messagebox.showerror("Error", "Failed to make prediction.")
                self.status_bar.config(text="Error during prediction")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            self.status_bar.config(text="Error during prediction")

    def display_results(self, result):
        """
        Display prediction results.
        
        Args:
            result (dict): Prediction result dictionary
        """
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        result_text = f"""
╔════════════════════════════════════════╗
║       PREDICTION RESULTS               ║
╚════════════════════════════════════════╝

Result: {result['prediction']}

Confidence: {result['confidence_percentage']:.2f}%

─ Probability Breakdown ─
Fracture Detected: {result['fracture_probability']:.4f} ({result['fracture_probability']*100:.2f}%)
No Fracture Detected: {result['normal_probability']:.4f} ({result['normal_probability']*100:.2f}%)

════════════════════════════════════════
"""
        
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)

    def display_preprocessing_steps(self, preprocessing_steps):
        """
        Display preprocessing steps.
        
        Args:
            preprocessing_steps (dict): Dictionary of preprocessing steps
        """
        # Clear existing tabs
        for tab in self.preprocessing_notebook.tabs():
            self.preprocessing_notebook.forget(tab)
        
        # Display each preprocessing step
        for step_name, step_image in preprocessing_steps.items():
            if step_image is not None and isinstance(step_image, np.ndarray):
                # Create a frame for this step
                frame = ttk.Frame(self.preprocessing_notebook)
                self.preprocessing_notebook.add(frame, text=step_name)
                
                # Convert to displayable format
                if step_name == 'original':
                    display_image = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                    display_image = Image.fromarray(display_image)
                elif len(step_image.shape) == 3:
                    display_image = Image.fromarray(step_image)
                else:
                    display_image = Image.fromarray(step_image)
                
                # Resize for display
                display_image.thumbnail((350, 350), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_image)
                
                canvas = tk.Canvas(frame, bg="gray")
                canvas.pack(fill=tk.BOTH, expand=True)
                canvas.create_image(
                    canvas.winfo_width()//2,
                    canvas.winfo_height()//2,
                    image=photo
                )
                canvas.image = photo

    def clear_all(self):
        """
        Clear all displays.
        """
        self.original_canvas.delete("all")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.current_image_path = None
        self.predict_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Cleared")

    def open_training_window(self):
        """
        Open the training window.
        """
        training_window = tk.Toplevel(self.root)
        training_window.title("Model Training")
        training_window.geometry("600x400")
        
        # Training frame
        frame = ttk.LabelFrame(training_window, text="Training Configuration", padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Epochs
        ttk.Label(frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(frame, from_=1, to=200, textvariable=epochs_var).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Batch size
        ttk.Label(frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        batch_var = tk.IntVar(value=32)
        ttk.Spinbox(frame, from_=8, to=128, textvariable=batch_var).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Data augmentation
        ttk.Label(frame, text="Data Augmentation:").grid(row=2, column=0, sticky=tk.W, pady=5)
        augment_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, variable=augment_var).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Progress
        progress_text = tk.Text(frame, height=10, width=60, state=tk.DISABLED)
        progress_text.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.NSEW)
        
        frame.grid_rowconfigure(3, weight=1)
        
        def start_training():
            epochs = epochs_var.get()
            batch_size = batch_var.get()
            
            # Disable button during training
            train_button.config(state=tk.DISABLED)
            progress_text.config(state=tk.NORMAL)
            progress_text.delete(1.0, tk.END)
            
            try:
                progress_text.insert(tk.END, "Initializing model...\n")
                progress_text.update()
                
                # Create model
                model = FractureDetectionModel()
                model.build_model()
                
                progress_text.insert(tk.END, "Loading data...\n")
                progress_text.update()
                
                # Load data
                data_loader = DataLoader(batch_size=batch_size)
                data_loader.load_data(augment=augment_var.get())
                
                progress_text.insert(tk.END, "Starting training...\n")
                progress_text.insert(tk.END, f"Epochs: {epochs}, Batch Size: {batch_size}\n")
                progress_text.update()
                
                # Create trainer
                trainer = ModelTrainer(model, data_loader)
                
                # Train model
                trainer.train(epochs=epochs, verbose=1)
                
                progress_text.insert(tk.END, "\nTraining completed!\n")
                progress_text.insert(tk.END, "Saving model...\n")
                progress_text.update()
                
                # Save model
                model.save_model()
                
                progress_text.insert(tk.END, "Model saved successfully!\n")
                progress_text.insert(tk.END, "\nEvaluating on test set...\n")
                progress_text.update()
                
                # Evaluate
                metrics = trainer.evaluate()
                trainer.print_evaluation_metrics()
                
                progress_text.insert(tk.END, "\nEvaluation complete!\n")
                progress_text.update()
                
                # Reload model in main app
                self.load_model()
                messagebox.showinfo("Success", "Model trained and saved successfully!")
            
            except Exception as e:
                progress_text.insert(tk.END, f"\nError: {e}\n")
                messagebox.showerror("Error", f"Training failed: {e}")
            
            finally:
                train_button.config(state=tk.NORMAL)
                progress_text.config(state=tk.DISABLED)
        
        # Train button
        train_button = ttk.Button(frame, text="Start Training", command=start_training)
        train_button.grid(row=4, column=0, columnspan=2, pady=10)


def main():
    """
    Main entry point of the application.
    """
    root = tk.Tk()
    app = FractureDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
